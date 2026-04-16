use fnp_dtype::ArrayStorage;
use fnp_iter::{Nditer, NditerOptions, NditerOrder};
use fnp_ndarray::{broadcast_shapes, element_count};
use fnp_ufunc::{UFuncArray, where_nonzero};
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

fn map_ufunc_error(err: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn extract_numeric_array(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    context: &str,
) -> PyResult<UFuncArray> {
    let numpy = py.import("numpy")?;
    let array = numpy.call_method1("asarray", (value,))?;
    let shape = array.getattr("shape")?.extract::<Vec<usize>>()?;
    let flat = array.call_method1("reshape", (-1,))?;
    let dtype = array.getattr("dtype")?;
    let dtype_name = dtype.str()?.extract::<String>()?;
    let kind = dtype.getattr("kind")?.extract::<String>()?;

    let storage = match kind.as_str() {
        "b" => ArrayStorage::Bool(flat.call_method0("tolist")?.extract::<Vec<bool>>()?),
        "i" => ArrayStorage::I64(
            flat.call_method1("astype", ("int64",))?
                .call_method0("tolist")?
                .extract::<Vec<i64>>()?,
        ),
        "u" => ArrayStorage::U64(
            flat.call_method1("astype", ("uint64",))?
                .call_method0("tolist")?
                .extract::<Vec<u64>>()?,
        ),
        "f" => ArrayStorage::F64(
            flat.call_method1("astype", ("float64",))?
                .call_method0("tolist")?
                .extract::<Vec<f64>>()?,
        ),
        _ => {
            return Err(PyTypeError::new_err(format!(
                "{context}: expected a bool/int/uint/float array, got dtype {dtype_name}",
            )));
        }
    };

    UFuncArray::from_storage(shape, storage).map_err(map_ufunc_error)
}

fn extract_index_vector(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    context: &str,
) -> PyResult<Vec<usize>> {
    let numpy = py.import("numpy")?;
    let array = numpy.call_method1("asarray", (value,))?;
    let flat = array.call_method1("reshape", (-1,))?;
    let indices = flat
        .call_method1("astype", ("int64",))?
        .call_method0("tolist")?
        .extract::<Vec<i64>>()?;

    indices
        .into_iter()
        .map(|index| {
            usize::try_from(index).map_err(|_| {
                PyValueError::new_err(format!(
                    "{context}: sorter indices must be non-negative, got {index}",
                ))
            })
        })
        .collect()
}

fn build_numpy_array_from_storage(
    py: Python<'_>,
    shape: &[usize],
    storage: ArrayStorage,
) -> PyResult<Py<PyAny>> {
    let numpy = py.import("numpy")?;
    let (list, dtype_name) = match storage {
        ArrayStorage::Bool(values) => {
            (PyList::new(py, values.iter().copied())?.into_any(), "bool_")
        }
        ArrayStorage::I8(values) => (PyList::new(py, values.iter().copied())?.into_any(), "int8"),
        ArrayStorage::I16(values) => (PyList::new(py, values.iter().copied())?.into_any(), "int16"),
        ArrayStorage::I32(values) => (PyList::new(py, values.iter().copied())?.into_any(), "int32"),
        ArrayStorage::I64(values) => (PyList::new(py, values.iter().copied())?.into_any(), "int64"),
        ArrayStorage::U8(values) => (PyList::new(py, values.iter().copied())?.into_any(), "uint8"),
        ArrayStorage::U16(values) => (
            PyList::new(py, values.iter().copied())?.into_any(),
            "uint16",
        ),
        ArrayStorage::U32(values) => (
            PyList::new(py, values.iter().copied())?.into_any(),
            "uint32",
        ),
        ArrayStorage::U64(values) => (
            PyList::new(py, values.iter().copied())?.into_any(),
            "uint64",
        ),
        ArrayStorage::F32(values) => (
            PyList::new(py, values.iter().copied())?.into_any(),
            "float32",
        ),
        ArrayStorage::F64(values) => (
            PyList::new(py, values.iter().copied())?.into_any(),
            "float64",
        ),
        unsupported => {
            return Err(PyTypeError::new_err(format!(
                "fnp-python: cannot export dtype {} to NumPy yet",
                unsupported.dtype().name()
            )));
        }
    };

    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", dtype_name)?;
    let array = numpy.call_method("array", (list,), Some(&kwargs))?;
    let output_shape = PyTuple::new(py, shape.iter().copied())?;
    Ok(array.call_method1("reshape", (&output_shape,))?.unbind())
}

fn build_numpy_array_from_ufunc(py: Python<'_>, array: &UFuncArray) -> PyResult<Py<PyAny>> {
    let storage = array.to_storage().map_err(map_ufunc_error)?;
    build_numpy_array_from_storage(py, array.shape(), storage)
}

fn build_numpy_tuple_from_ufuncs(py: Python<'_>, arrays: &[UFuncArray]) -> PyResult<Py<PyAny>> {
    let arrays = arrays
        .iter()
        .map(|array| build_numpy_array_from_ufunc(py, array))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, arrays.iter().map(|array| array.bind(py)))?
        .into_any()
        .unbind())
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

#[pyfunction]
#[pyo3(signature = (x, bins, right=false))]
fn digitize(py: Python<'_>, x: Py<PyAny>, bins: Py<PyAny>, right: bool) -> PyResult<Py<PyAny>> {
    let x = extract_numeric_array(py, x.bind(py), "digitize(x)")?;
    let bins = extract_numeric_array(py, bins.bind(py), "digitize(bins)")?;
    let result = x.digitize_right(&bins, right).map_err(map_ufunc_error)?;
    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction]
#[pyo3(signature = (x, xp, fp, left=None, right=None))]
fn interp(
    py: Python<'_>,
    x: Py<PyAny>,
    xp: Py<PyAny>,
    fp: Py<PyAny>,
    left: Option<f64>,
    right: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let x = extract_numeric_array(py, x.bind(py), "interp(x)")?;
    let xp = extract_numeric_array(py, xp.bind(py), "interp(xp)")?;
    let fp = extract_numeric_array(py, fp.bind(py), "interp(fp)")?;
    let result = UFuncArray::interp_lr(&x, &xp, &fp, left, right).map_err(map_ufunc_error)?;
    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction(name = "where")]
#[pyo3(signature = (condition, x=None, y=None))]
fn where_py(
    py: Python<'_>,
    condition: Py<PyAny>,
    x: Option<Py<PyAny>>,
    y: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let condition = extract_numeric_array(py, condition.bind(py), "where(condition)")?;

    match (x, y) {
        (Some(x), Some(y)) => {
            let x = extract_numeric_array(py, x.bind(py), "where(x)")?;
            let y = extract_numeric_array(py, y.bind(py), "where(y)")?;
            let result = UFuncArray::where_select(&condition, &x, &y).map_err(map_ufunc_error)?;
            build_numpy_array_from_ufunc(py, &result)
        }
        (None, None) => {
            let result = where_nonzero(&condition).map_err(map_ufunc_error)?;
            build_numpy_tuple_from_ufuncs(py, &result)
        }
        _ => Err(PyValueError::new_err(
            "where: either provide both x and y or neither",
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (a, v, side="left", sorter=None))]
fn searchsorted(
    py: Python<'_>,
    a: Py<PyAny>,
    v: Py<PyAny>,
    side: &str,
    sorter: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let a = extract_numeric_array(py, a.bind(py), "searchsorted(a)")?;
    let v = extract_numeric_array(py, v.bind(py), "searchsorted(v)")?;
    let sorter = sorter
        .map(|sorter| extract_index_vector(py, sorter.bind(py), "searchsorted(sorter)"))
        .transpose()?;
    let result = a
        .searchsorted(&v, Some(side), sorter.as_deref())
        .map_err(map_ufunc_error)?;
    build_numpy_array_from_ufunc(py, &result)
}

#[pymodule]
fn fnp_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNditerStep>()?;
    m.add_class::<PyNditer>()?;
    m.add_class::<PyFromPyFunc>()?;
    m.add_class::<PyVectorize>()?;
    m.add_function(wrap_pyfunction!(frompyfunc, m)?)?;
    m.add_function(wrap_pyfunction!(vectorize, m)?)?;
    m.add_function(wrap_pyfunction!(digitize, m)?)?;
    m.add_function(wrap_pyfunction!(interp, m)?)?;
    m.add_function(wrap_pyfunction!(where_py, m)?)?;
    m.add_function(wrap_pyfunction!(searchsorted, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{PyFromPyFunc, PyVectorize, digitize, fnp_python, interp, searchsorted, where_py};
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

    fn numeric_array<'py>(
        py: Python<'py>,
        values: impl IntoPyObject<'py>,
        dtype: &str,
    ) -> pyo3::Bound<'py, pyo3::types::PyAny> {
        let numpy = py.import("numpy").unwrap();
        let kwargs = PyDict::new(py);
        kwargs
            .set_item("dtype", dtype)
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
            assert!(module.getattr("digitize").is_ok());
            assert!(module.getattr("interp").is_ok());
            assert!(module.getattr("where").is_ok());
            assert!(module.getattr("searchsorted").is_ok());
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

    #[test]
    fn digitize_matches_numpy_increasing_bins() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let values = numeric_array(py, vec![0.2, 6.4, 3.0, 1.6], "float64");
            let bins = numeric_array(py, vec![0.0, 1.0, 2.5, 4.0, 10.0], "float64");

            let actual = digitize(py, values.clone().unbind(), bins.clone().unbind(), false)?;
            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("digitize")?.call1((values, bins))?;

            assert_eq!(
                actual
                    .bind(py)
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?,
                "int64"
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn digitize_matches_numpy_right_true_for_decreasing_bins() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let values = numeric_array(py, vec![9.0, 5.0, 0.5], "float64");
            let bins = numeric_array(py, vec![10.0, 6.0, 3.0, 0.0], "float64");

            let actual = digitize(py, values.clone().unbind(), bins.clone().unbind(), true)?;
            let numpy = py.import("numpy")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("right", true)?;
            let expected = numpy.call_method("digitize", (values, bins), Some(&kwargs))?;

            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn digitize_preserves_large_uint64_bins() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let large = 9_007_199_254_740_993_u64;
            let values = numeric_array(py, vec![large, large + 1], "uint64");
            let bins = numeric_array(py, vec![large - 1, large, large + 2], "uint64");

            let actual = digitize(py, values.clone().unbind(), bins.clone().unbind(), false)?;
            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("digitize")?.call1((values, bins))?;

            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn interp_matches_numpy_with_multidimensional_x() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let x = numeric_array(py, vec![vec![0.5, 1.5], vec![2.5, 3.5]], "float64");
            let xp = numeric_array(py, vec![0.0, 1.0, 2.0, 4.0], "float64");
            let fp = numeric_array(py, vec![0.0, 10.0, 20.0, 40.0], "float64");

            let actual = interp(
                py,
                x.clone().unbind(),
                xp.clone().unbind(),
                fp.clone().unbind(),
                None,
                None,
            )?;
            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("interp")?.call1((x, xp, fp))?;

            assert_eq!(
                actual.bind(py).getattr("shape")?.extract::<Vec<usize>>()?,
                vec![2, 2]
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn interp_matches_numpy_with_custom_left_and_right() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let x = numeric_array(py, vec![-1.0, 0.0, 3.0, 5.0], "float64");
            let xp = numeric_array(py, vec![0.0, 1.0, 4.0], "float64");
            let fp = numeric_array(py, vec![10.0, 20.0, 30.0], "float64");

            let actual = interp(
                py,
                x.clone().unbind(),
                xp.clone().unbind(),
                fp.clone().unbind(),
                Some(-5.0),
                Some(99.0),
            )?;
            let numpy = py.import("numpy")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("left", -5.0)?;
            kwargs.set_item("right", 99.0)?;
            let expected = numpy.call_method("interp", (x, xp, fp), Some(&kwargs))?;

            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn interp_accepts_integer_xp_and_fp_inputs() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let x = numeric_array(py, vec![0_i64, 2, 3], "int64");
            let xp = numeric_array(py, vec![0_i64, 1, 4], "int64");
            let fp = numeric_array(py, vec![5_i64, 15, 45], "int64");

            let actual = interp(
                py,
                x.clone().unbind(),
                xp.clone().unbind(),
                fp.clone().unbind(),
                None,
                None,
            )?;
            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("interp")?.call1((x, xp, fp))?;

            assert_eq!(
                actual
                    .bind(py)
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?,
                "float64"
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn where_three_arg_matches_numpy_broadcasting() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let condition =
                numeric_array(py, vec![vec![1_i64, 0_i64], vec![0_i64, 1_i64]], "int64");
            let x = numeric_array(py, vec![10.0, 20.0], "float64");
            let y = numeric_array(py, vec![vec![1.0], vec![2.0]], "float64");

            let actual = where_py(
                py,
                condition.clone().unbind(),
                Some(x.clone().unbind()),
                Some(y.clone().unbind()),
            )?;
            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("where")?.call1((condition, x, y))?;

            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn where_three_arg_preserves_large_uint64_values() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let large = 9_007_199_254_740_993_u64;
            let condition = numeric_array(py, vec![1_i64, 0_i64, 1_i64], "int64");
            let x = numeric_array(py, vec![large, large + 1, large + 2], "uint64");
            let y = numeric_array(py, vec![7_u64, 8_u64, 9_u64], "uint64");

            let actual = where_py(
                py,
                condition.clone().unbind(),
                Some(x.clone().unbind()),
                Some(y.clone().unbind()),
            )?;
            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("where")?.call1((condition, x, y))?;

            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn where_single_argument_matches_numpy_nonzero_indices() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let condition = numeric_array(
                py,
                vec![vec![0_i64, 1_i64, 0_i64], vec![2_i64, 0_i64, 3_i64]],
                "int64",
            );

            let actual = where_py(py, condition.clone().unbind(), None, None)?;
            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("where")?.call1((condition,))?;

            let actual = actual.bind(py).downcast::<PyTuple>()?;
            let expected = expected.downcast::<PyTuple>()?;
            assert_eq!(actual.len()?, 2);
            assert_eq!(expected.len()?, 2);

            for (actual_item, expected_item) in actual.try_iter()?.zip(expected.try_iter()?) {
                let actual_item = actual_item?;
                let expected_item = expected_item?;
                assert_eq!(
                    repr_string(&actual_item.call_method0("tolist")?),
                    repr_string(&expected_item.call_method0("tolist")?)
                );
            }
            Ok(())
        });
    }

    #[test]
    fn searchsorted_matches_numpy_left_side_with_duplicates() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let sorted = numeric_array(py, vec![1.0, 2.0, 2.0, 2.0, 3.0], "float64");
            let values = numeric_array(py, vec![0.0, 2.0, 2.5, 4.0], "float64");

            let actual = searchsorted(
                py,
                sorted.clone().unbind(),
                values.clone().unbind(),
                "left",
                None,
            )?;
            let numpy = py.import("numpy")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("side", "left")?;
            let expected = numpy.call_method("searchsorted", (sorted, values), Some(&kwargs))?;

            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn searchsorted_matches_numpy_right_side_with_probe_shape() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let sorted = numeric_array(py, vec![1.0, 3.0, 5.0, 7.0], "float64");
            let values = numeric_array(py, vec![vec![0.0, 4.0], vec![6.0, 8.0]], "float64");

            let actual = searchsorted(
                py,
                sorted.clone().unbind(),
                values.clone().unbind(),
                "right",
                None,
            )?;
            let numpy = py.import("numpy")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("side", "right")?;
            let expected = numpy.call_method("searchsorted", (sorted, values), Some(&kwargs))?;

            assert_eq!(
                actual.bind(py).getattr("shape")?.extract::<Vec<usize>>()?,
                vec![2, 2]
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn searchsorted_supports_sorter_argument() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let values = numeric_array(py, vec![30_u64, 10_u64, 20_u64], "uint64");
            let probes = numeric_array(py, vec![15_u64, 30_u64], "uint64");
            let sorter = numeric_array(py, vec![1_i64, 2_i64, 0_i64], "int64");

            let actual = searchsorted(
                py,
                values.clone().unbind(),
                probes.clone().unbind(),
                "left",
                Some(sorter.clone().unbind()),
            )?;
            let numpy = py.import("numpy")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("sorter", sorter)?;
            let expected = numpy.call_method("searchsorted", (values, probes), Some(&kwargs))?;

            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }
}
