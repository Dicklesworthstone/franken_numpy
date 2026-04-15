use fnp_iter::{Nditer, NditerOptions, NditerOrder};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::Bound;

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

#[pymodule]
fn fnp_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNditerStep>()?;
    m.add_class::<PyNditer>()?;
    Ok(())
}
