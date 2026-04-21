use fnp_dtype::{ArrayStorage, DType, f16, promote};
use fnp_iter::{Nditer, NditerOptions, NditerOrder};
use fnp_linalg::{pinv_hermitian_nxn_with_tolerance_aliases, pinv_mxn_with_tolerance_aliases};
use fnp_ndarray::{broadcast_shapes, element_count};
use fnp_ufunc::{
    FromPyFuncReduceAxisSpec, FromPyFuncReduceError, FromPyFuncReduceIdentity,
    FromPyFuncReduceOptions, GridSpec, UFuncArray, UnaryOp, copysign as ufunc_copysign,
    frexp as ufunc_frexp, hypot as ufunc_hypot, isneginf as ufunc_isneginf,
    isposinf as ufunc_isposinf, ldexp as ufunc_ldexp, logaddexp as ufunc_logaddexp,
    logaddexp2 as ufunc_logaddexp2, modf as ufunc_modf, nextafter as ufunc_nextafter,
    reduce_frompyfunc_values, signbit as ufunc_signbit, spacing as ufunc_spacing, where_nonzero,
};
use pyo3::Bound;
use pyo3::exceptions::{PyTypeError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyList, PyModule, PyTuple};
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
    display_name: String,
    identity: FromPyFuncReduceIdentity<Py<PyAny>>,
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

#[pyclass(name = "MGridClass", unsendable)]
pub struct PyMGridClass;

#[pyclass(name = "OGridClass", unsendable)]
pub struct PyOGridClass;

#[pyclass(name = "RClass", unsendable)]
pub struct PyRClass;

#[pyclass(name = "CClass", unsendable)]
pub struct PyCClass;

#[derive(Clone, Copy)]
enum AxisConcatenatorKind {
    Rows,
    Columns,
}

impl AxisConcatenatorKind {
    const fn context(self) -> &'static str {
        match self {
            Self::Rows => "r_",
            Self::Columns => "c_",
        }
    }

    fn concatenate(self, arrays: &[UFuncArray]) -> Result<UFuncArray, fnp_ufunc::UFuncError> {
        match self {
            Self::Rows => UFuncArray::r_(arrays),
            Self::Columns => UFuncArray::c_(arrays),
        }
    }
}

#[derive(Clone, Copy)]
enum StackHelperKind {
    Vertical,
    Horizontal,
    Depth,
    Column,
}

impl StackHelperKind {
    const fn context(self) -> &'static str {
        match self {
            Self::Vertical => "vstack",
            Self::Horizontal => "hstack",
            Self::Depth => "dstack",
            Self::Column => "column_stack",
        }
    }

    fn rust_default(self, arrays: &[UFuncArray]) -> Result<UFuncArray, fnp_ufunc::UFuncError> {
        match self {
            Self::Vertical => UFuncArray::vstack(arrays),
            Self::Horizontal => UFuncArray::hstack(arrays),
            Self::Depth => UFuncArray::dstack(arrays),
            Self::Column => UFuncArray::column_stack(arrays),
        }
    }
}

#[derive(Clone, Copy)]
enum SplitHelperKind {
    Equal,
    Flexible,
    Horizontal,
    Vertical,
    Depth,
}

impl SplitHelperKind {
    const fn context(self) -> &'static str {
        match self {
            Self::Equal => "split",
            Self::Flexible => "array_split",
            Self::Horizontal => "hsplit",
            Self::Vertical => "vsplit",
            Self::Depth => "dsplit",
        }
    }

    fn rust_sections(
        self,
        array: &UFuncArray,
        sections: usize,
        axis: isize,
    ) -> Result<Vec<UFuncArray>, fnp_ufunc::UFuncError> {
        match self {
            Self::Equal => array.split(sections, axis),
            Self::Flexible => array.array_split(sections, axis),
            Self::Horizontal => array.hsplit(sections),
            Self::Vertical => array.vsplit(sections),
            Self::Depth => array.dsplit(sections),
        }
    }
}

struct ParsedGridAxis {
    spec: GridSpec,
    start: Py<PyAny>,
    stop: Py<PyAny>,
    step_for_dtype: Py<PyAny>,
}

fn map_ufunc_error(err: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[derive(Clone, Copy)]
enum OptionalFloatKwarg {
    Omitted,
    None,
    Value(f64),
}

impl OptionalFloatKwarg {
    fn parse(py: Python<'_>, value: Option<Py<PyAny>>, name: &str) -> PyResult<Self> {
        match value {
            None => Ok(Self::Omitted),
            Some(value) if value.bind(py).is_none() => Ok(Self::None),
            Some(value) => Ok(Self::Value(value.bind(py).extract::<f64>().map_err(
                |_| PyTypeError::new_err(format!("pinv: {name} must be a real number or None")),
            )?)),
        }
    }

    fn as_rcond(self) -> Option<f64> {
        match self {
            Self::Value(value) => Some(value),
            Self::Omitted | Self::None => None,
        }
    }

    fn as_rtol(self) -> Option<Option<f64>> {
        match self {
            Self::Omitted => None,
            Self::None => Some(None),
            Self::Value(value) => Some(Some(value)),
        }
    }

    fn set_on_kwargs(self, kwargs: &Bound<'_, PyDict>, name: &str) -> PyResult<()> {
        match self {
            Self::Omitted => Ok(()),
            Self::None => kwargs.set_item(name, kwargs.py().None()),
            Self::Value(value) => kwargs.set_item(name, value),
        }
    }
}

fn parse_pinv_rtol_kwarg(
    py: Python<'_>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<OptionalFloatKwarg> {
    let Some(kwargs) = kwargs else {
        return Ok(OptionalFloatKwarg::Omitted);
    };

    let mut rtol = OptionalFloatKwarg::Omitted;
    for (key, value) in kwargs.iter() {
        let name = key.extract::<String>()?;
        match name.as_str() {
            "rtol" => {
                rtol = OptionalFloatKwarg::parse(py, Some(value.unbind()), "rtol")?;
            }
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "pinv() got an unexpected keyword argument '{name}'"
                )));
            }
        }
    }

    Ok(rtol)
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

fn extract_precise_numeric_array(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    context: &str,
) -> PyResult<UFuncArray> {
    let numpy = py.import("numpy")?;
    let array = numpy.call_method1("asarray", (value,))?;
    let shape = array.getattr("shape")?.extract::<Vec<usize>>()?;
    let flat = array.call_method1("reshape", (-1,))?;
    let dtype = array.getattr("dtype")?;
    let dtype_name = dtype.getattr("name")?.extract::<String>()?;
    let parsed_dtype = DType::parse(&dtype_name).ok_or_else(|| {
        PyTypeError::new_err(format!(
            "{context}: expected a bool/int/uint/float array, got dtype {dtype_name}",
        ))
    })?;

    let storage = match parsed_dtype {
        DType::Bool => ArrayStorage::Bool(flat.call_method0("tolist")?.extract::<Vec<bool>>()?),
        DType::I8 => ArrayStorage::I8(
            flat.call_method1("astype", ("int8",))?
                .call_method0("tolist")?
                .extract::<Vec<i8>>()?,
        ),
        DType::I16 => ArrayStorage::I16(
            flat.call_method1("astype", ("int16",))?
                .call_method0("tolist")?
                .extract::<Vec<i16>>()?,
        ),
        DType::I32 => ArrayStorage::I32(
            flat.call_method1("astype", ("int32",))?
                .call_method0("tolist")?
                .extract::<Vec<i32>>()?,
        ),
        DType::I64 => ArrayStorage::I64(
            flat.call_method1("astype", ("int64",))?
                .call_method0("tolist")?
                .extract::<Vec<i64>>()?,
        ),
        DType::U8 => ArrayStorage::U8(
            flat.call_method1("astype", ("uint8",))?
                .call_method0("tolist")?
                .extract::<Vec<u8>>()?,
        ),
        DType::U16 => ArrayStorage::U16(
            flat.call_method1("astype", ("uint16",))?
                .call_method0("tolist")?
                .extract::<Vec<u16>>()?,
        ),
        DType::U32 => ArrayStorage::U32(
            flat.call_method1("astype", ("uint32",))?
                .call_method0("tolist")?
                .extract::<Vec<u32>>()?,
        ),
        DType::U64 => ArrayStorage::U64(
            flat.call_method1("astype", ("uint64",))?
                .call_method0("tolist")?
                .extract::<Vec<u64>>()?,
        ),
        DType::F16 => ArrayStorage::F16(
            flat.call_method1("astype", ("float16",))?
                .call_method0("tolist")?
                .extract::<Vec<f32>>()?
                .into_iter()
                .map(f16::from_f32)
                .collect(),
        ),
        DType::F32 => ArrayStorage::F32(
            flat.call_method1("astype", ("float32",))?
                .call_method0("tolist")?
                .extract::<Vec<f32>>()?,
        ),
        DType::F64 => ArrayStorage::F64(
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

fn extract_integer_array(
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
        _ => {
            return Err(PyTypeError::new_err(format!(
                "{context}: expected an integer index array, got dtype {dtype_name}",
            )));
        }
    };

    UFuncArray::from_storage(shape, storage).map_err(map_ufunc_error)
}

fn extract_index_shape(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    context: &str,
) -> PyResult<Vec<usize>> {
    let numpy = py.import("numpy")?;
    let array = numpy.call_method1("asarray", (value,))?;
    let ndim = array.getattr("ndim")?.extract::<usize>()?;

    if ndim == 0 {
        let dim = array.extract::<i64>().map_err(|_| {
            PyTypeError::new_err(format!(
                "{context}: shape entries must be integers, not {}",
                array
                    .getattr("dtype")
                    .and_then(|dtype| dtype.str())
                    .and_then(|name| name.extract::<String>())
                    .unwrap_or_else(|_| "unknown".to_string())
            ))
        })?;
        return usize::try_from(dim).map(|dim| vec![dim]).map_err(|_| {
            PyValueError::new_err(format!("{context}: shape entries must be non-negative"))
        });
    }

    let flat = array.call_method1("reshape", (-1,))?;
    if flat.getattr("size")?.extract::<usize>()? == 0 {
        return Ok(vec![]);
    }

    let dtype = flat.getattr("dtype")?;
    let kind = dtype.getattr("kind")?.extract::<String>()?;
    match kind.as_str() {
        "i" => flat
            .call_method1("astype", ("int64",))?
            .call_method0("tolist")?
            .extract::<Vec<i64>>()?
            .into_iter()
            .map(|dim| {
                usize::try_from(dim).map_err(|_| {
                    PyValueError::new_err(format!("{context}: shape entries must be non-negative"))
                })
            })
            .collect(),
        "u" => flat
            .call_method1("astype", ("uint64",))?
            .call_method0("tolist")?
            .extract::<Vec<u64>>()?
            .into_iter()
            .map(|dim| {
                usize::try_from(dim).map_err(|_| {
                    PyValueError::new_err(format!(
                        "{context}: shape entry {dim} exceeds platform usize"
                    ))
                })
            })
            .collect(),
        _ => Err(PyTypeError::new_err(format!(
            "{context}: shape entries must be integers, not {}",
            dtype.str()?.extract::<String>()?
        ))),
    }
}

fn extract_take_indices(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    context: &str,
) -> PyResult<(Vec<usize>, Vec<i64>)> {
    let indices = extract_integer_array(py, value, context)?;
    let shape = indices.shape().to_vec();
    let storage = indices.to_storage().map_err(map_ufunc_error)?;

    let indices = match storage {
        ArrayStorage::I64(values) => values,
        ArrayStorage::U64(values) => values
            .into_iter()
            .map(|value| {
                i64::try_from(value).map_err(|_| {
                    PyValueError::new_err(format!(
                        "{context}: index {value} exceeds signed 64-bit range",
                    ))
                })
            })
            .collect::<PyResult<Vec<_>>>()?,
        _ => unreachable!("extract_integer_array only produces signed/unsigned integer storage"),
    };

    Ok((shape, indices))
}

fn extract_condition_mask(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    context: &str,
) -> PyResult<Vec<bool>> {
    let mask = extract_numeric_array(py, value, context)?;
    Ok(mask.values().iter().map(|&value| value != 0.0).collect())
}

fn extract_ravel_multi_index_inputs(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    ndim: usize,
) -> PyResult<Vec<UFuncArray>> {
    let iter = value.try_iter().map_err(|_| {
        PyValueError::new_err(format!(
            "parameter multi_index must be a sequence of length {ndim}"
        ))
    })?;
    let arrays = iter
        .enumerate()
        .map(|(index, item)| {
            let item = item?;
            extract_integer_array(
                py,
                &item,
                &format!("ravel_multi_index(multi_index[{index}])"),
            )
        })
        .collect::<PyResult<Vec<_>>>()?;

    if arrays.len() != ndim {
        return Err(PyValueError::new_err(format!(
            "parameter multi_index must be a sequence of length {ndim}"
        )));
    }
    Ok(arrays)
}

fn extract_ravel_multi_index_modes(
    py: Python<'_>,
    value: Option<Py<PyAny>>,
) -> PyResult<Vec<String>> {
    let Some(value) = value else {
        return Ok(vec!["raise".to_string()]);
    };

    let value = value.bind(py);
    if let Ok(mode) = value.extract::<String>() {
        return Ok(vec![mode]);
    }

    if let Ok(iter) = value.try_iter() {
        return iter
            .enumerate()
            .map(|(index, item)| {
                item?.extract::<String>().map_err(|_| {
                    PyTypeError::new_err(format!(
                        "ravel_multi_index(mode): mode[{index}] must be a string"
                    ))
                })
            })
            .collect();
    }

    Err(PyTypeError::new_err(
        "ravel_multi_index(mode): mode must be a string or sequence of strings",
    ))
}

fn extract_numeric_array_sequence(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    context: &str,
) -> PyResult<Vec<UFuncArray>> {
    value
        .try_iter()?
        .enumerate()
        .map(|(index, item)| {
            let item = item?;
            extract_numeric_array(py, &item, &format!("{context}[{index}]"))
        })
        .collect()
}

fn extract_stack_sequence_items(
    value: &Bound<'_, PyAny>,
    context: &str,
) -> PyResult<Vec<Py<PyAny>>> {
    if !value.hasattr("__getitem__")? {
        return Err(PyTypeError::new_err(format!(
            "{context}: arrays to stack must be passed as a \"sequence\" type such as list or tuple."
        )));
    }

    value.try_iter()?.map(|item| Ok(item?.unbind())).collect()
}

fn extract_stack_numeric_arrays(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    kind: StackHelperKind,
) -> PyResult<Option<Vec<UFuncArray>>> {
    let items = extract_stack_sequence_items(value, kind.context())?;
    let mut arrays = Vec::with_capacity(items.len());

    for (index, item) in items.into_iter().enumerate() {
        match extract_precise_numeric_array(
            py,
            item.bind(py),
            &format!("{}(tup[{index}])", kind.context()),
        ) {
            Ok(array) => arrays.push(array),
            Err(_) => return Ok(None),
        }
    }

    Ok(Some(arrays))
}

fn stack_helper_numpy_fallback(
    py: Python<'_>,
    kind: StackHelperKind,
    tup: Py<PyAny>,
    dtype: Option<Py<PyAny>>,
    casting: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let numpy = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    let mut has_kwargs = false;

    if let Some(dtype) = dtype
        && !dtype.bind(py).is_none()
    {
        kwargs.set_item("dtype", dtype.bind(py))?;
        has_kwargs = true;
    }
    if let Some(casting) = casting {
        kwargs.set_item("casting", casting)?;
        has_kwargs = true;
    }

    let result = if has_kwargs {
        numpy
            .getattr(kind.context())?
            .call((tup.bind(py),), Some(&kwargs))?
    } else {
        numpy.getattr(kind.context())?.call1((tup.bind(py),))?
    };
    Ok(result.unbind())
}

fn stack_helper_default(
    py: Python<'_>,
    tup: Py<PyAny>,
    kind: StackHelperKind,
) -> PyResult<Py<PyAny>> {
    let Some(arrays) = extract_stack_numeric_arrays(py, tup.bind(py), kind)? else {
        return stack_helper_numpy_fallback(py, kind, tup, None, None);
    };

    let result = kind.rust_default(&arrays).map_err(map_ufunc_error)?;
    build_numpy_array_from_ufunc(py, &result)
}

fn extract_split_sections(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
    let numpy = py.import("numpy")?;
    let array = numpy.call_method1("asarray", (value,))?;
    if array.getattr("ndim")?.extract::<usize>()? != 0 {
        return Ok(None);
    }

    let kind = array
        .getattr("dtype")?
        .getattr("kind")?
        .extract::<String>()?;
    if !matches!(kind.as_str(), "b" | "i" | "u" | "f") {
        return Ok(None);
    }

    let sections = array
        .call_method1("astype", ("int64",))?
        .call_method0("item")?
        .extract::<i64>()?;
    if sections < 0 {
        return Err(PyValueError::new_err(
            "number sections must be larger than 0.",
        ));
    }

    usize::try_from(sections)
        .map(Some)
        .map_err(|_| PyValueError::new_err("number sections must be larger than 0."))
}

fn split_helper_numpy_fallback(
    py: Python<'_>,
    kind: SplitHelperKind,
    ary: Py<PyAny>,
    indices_or_sections: Py<PyAny>,
    axis: Option<isize>,
) -> PyResult<Py<PyAny>> {
    let numpy = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    let has_kwargs = if let Some(axis) = axis {
        kwargs.set_item("axis", axis)?;
        true
    } else {
        false
    };

    let result = if has_kwargs {
        numpy
            .getattr(kind.context())?
            .call((ary.bind(py), indices_or_sections.bind(py)), Some(&kwargs))?
    } else {
        numpy
            .getattr(kind.context())?
            .call1((ary.bind(py), indices_or_sections.bind(py)))?
    };
    Ok(result.unbind())
}

fn split_helper_default(
    py: Python<'_>,
    ary: Py<PyAny>,
    indices_or_sections: Py<PyAny>,
    kind: SplitHelperKind,
    axis: Option<isize>,
) -> PyResult<Py<PyAny>> {
    let Some(sections) = extract_split_sections(py, indices_or_sections.bind(py))? else {
        return split_helper_numpy_fallback(py, kind, ary, indices_or_sections, axis);
    };

    if sections == 0 {
        return match kind {
            SplitHelperKind::Flexible => Err(PyValueError::new_err(
                "number sections must be larger than 0.",
            )),
            _ => Err(PyZeroDivisionError::new_err("integer modulo by zero")),
        };
    }

    let array = match extract_precise_numeric_array(py, ary.bind(py), kind.context()) {
        Ok(array) => array,
        Err(_) => return split_helper_numpy_fallback(py, kind, ary, indices_or_sections, axis),
    };

    let result = kind
        .rust_sections(&array, sections, axis.unwrap_or(0))
        .map_err(map_ufunc_error)?;
    build_numpy_list_from_ufuncs(py, &result)
}

fn extract_axis_spec(
    py: Python<'_>,
    axis: Option<Py<PyAny>>,
    context: &str,
) -> PyResult<Option<Vec<isize>>> {
    let Some(axis) = axis else {
        return Ok(None);
    };

    let axis = axis.bind(py);
    if axis.is_none() {
        return Ok(None);
    }

    if let Ok(axis) = axis.extract::<isize>() {
        return Ok(Some(vec![axis]));
    }

    if let Ok(iter) = axis.try_iter() {
        let axes = iter
            .enumerate()
            .map(|(index, item)| {
                item?.extract::<isize>().map_err(|_| {
                    PyTypeError::new_err(format!("{context}: axis[{index}] must be an integer"))
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(Some(axes));
    }

    Err(PyTypeError::new_err(format!(
        "{context}: axis must be an integer, tuple/list of integers, or None",
    )))
}

#[allow(dead_code)]
fn extract_tensorsolve_axes(
    py: Python<'_>,
    axes: Option<Py<PyAny>>,
    ndim: usize,
) -> PyResult<Option<Vec<usize>>> {
    let Some(axes) = axes else {
        return Ok(None);
    };

    let axes = axes.bind(py);
    if axes.is_none() {
        return Ok(None);
    }

    let iter = axes.try_iter()?;
    let permutation = PyList::new(py, 0..ndim)?;
    for axis in iter {
        let axis = axis?;
        permutation.call_method1("remove", (axis.clone(),))?;
        permutation.call_method1("insert", (ndim, axis))?;
    }

    let mut normalized = Vec::with_capacity(ndim);
    for axis in permutation.iter() {
        if axis.downcast::<PyBool>().is_ok() {
            return Err(PyTypeError::new_err("an integer is required"));
        }
        match axis.extract::<usize>() {
            Ok(value) => normalized.push(value),
            Err(_) => {
                let type_name = axis.get_type().name()?;
                return Err(PyTypeError::new_err(format!(
                    "'{type_name}' object cannot be interpreted as an integer"
                )));
            }
        }
    }

    Ok(Some(normalized))
}

fn require_numpy_ndarray(py: Python<'_>, value: &Bound<'_, PyAny>, context: &str) -> PyResult<()> {
    let builtins = py.import("builtins")?;
    let numpy = py.import("numpy")?;
    let is_ndarray = builtins
        .call_method1("isinstance", (value, numpy.getattr("ndarray")?))?
        .extract::<bool>()?;

    if is_ndarray {
        Ok(())
    } else {
        Err(PyTypeError::new_err(format!(
            "{context}: argument 1 must be numpy.ndarray",
        )))
    }
}

fn reshape_with_leading_singletons(
    array: UFuncArray,
    target_ndim: usize,
    context: &str,
) -> PyResult<UFuncArray> {
    if array.shape().len() >= target_ndim {
        return Ok(array);
    }

    let mut reshaped = Vec::with_capacity(target_ndim);
    reshaped.extend(std::iter::repeat_n(
        1_isize,
        target_ndim - array.shape().len(),
    ));
    reshaped.extend(
        array
            .shape()
            .iter()
            .map(|&dim| {
                isize::try_from(dim).map_err(|_| {
                    PyValueError::new_err(format!(
                        "{context}: dimension {dim} exceeds signed pointer range",
                    ))
                })
            })
            .collect::<PyResult<Vec<_>>>()?,
    );

    array
        .reshape(&reshaped)
        .map_err(|err| map_ufunc_error(format!("{context}: {err}")))
}

fn copy_result_into_numpy_array(
    py: Python<'_>,
    target: &Bound<'_, PyAny>,
    array: &UFuncArray,
) -> PyResult<()> {
    let numpy = py.import("numpy")?;
    let updated = build_numpy_array_from_ufunc(py, array)?;
    numpy.call_method1("copyto", (target, updated.bind(py)))?;
    Ok(())
}

fn extract_array_shape(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    context: &str,
) -> PyResult<Vec<usize>> {
    let numpy = py.import("numpy")?;
    let array = numpy.call_method1("asarray", (value,))?;
    array
        .getattr("shape")?
        .extract::<Vec<usize>>()
        .map_err(|err| PyTypeError::new_err(format!("{context}: could not read shape: {err}")))
}

fn extract_python_dtype(
    py: Python<'_>,
    dtype: Option<Py<PyAny>>,
    default: DType,
    context: &str,
) -> PyResult<DType> {
    let Some(dtype) = dtype else {
        return Ok(default);
    };

    let dtype = dtype.bind(py);
    if dtype.is_none() {
        return Ok(default);
    }

    let numpy = py.import("numpy")?;
    let parsed = numpy.getattr("dtype")?.call1((dtype,))?;
    let name = parsed.getattr("name")?.extract::<String>()?;
    DType::parse(&name)
        .ok_or_else(|| PyTypeError::new_err(format!("{context}: unsupported dtype {name}")))
}

#[allow(dead_code)]
fn extract_storage_from_flat_array(
    flat: &Bound<'_, PyAny>,
    parsed_dtype: DType,
    context: &str,
) -> PyResult<ArrayStorage> {
    match parsed_dtype {
        DType::Bool => Ok(ArrayStorage::Bool(
            flat.call_method0("tolist")?.extract::<Vec<bool>>()?,
        )),
        DType::I8 => Ok(ArrayStorage::I8(
            flat.call_method1("astype", ("int8",))?
                .call_method0("tolist")?
                .extract::<Vec<i8>>()?,
        )),
        DType::I16 => Ok(ArrayStorage::I16(
            flat.call_method1("astype", ("int16",))?
                .call_method0("tolist")?
                .extract::<Vec<i16>>()?,
        )),
        DType::I32 => Ok(ArrayStorage::I32(
            flat.call_method1("astype", ("int32",))?
                .call_method0("tolist")?
                .extract::<Vec<i32>>()?,
        )),
        DType::I64 => Ok(ArrayStorage::I64(
            flat.call_method1("astype", ("int64",))?
                .call_method0("tolist")?
                .extract::<Vec<i64>>()?,
        )),
        DType::U8 => Ok(ArrayStorage::U8(
            flat.call_method1("astype", ("uint8",))?
                .call_method0("tolist")?
                .extract::<Vec<u8>>()?,
        )),
        DType::U16 => Ok(ArrayStorage::U16(
            flat.call_method1("astype", ("uint16",))?
                .call_method0("tolist")?
                .extract::<Vec<u16>>()?,
        )),
        DType::U32 => Ok(ArrayStorage::U32(
            flat.call_method1("astype", ("uint32",))?
                .call_method0("tolist")?
                .extract::<Vec<u32>>()?,
        )),
        DType::U64 => Ok(ArrayStorage::U64(
            flat.call_method1("astype", ("uint64",))?
                .call_method0("tolist")?
                .extract::<Vec<u64>>()?,
        )),
        DType::F16 => Ok(ArrayStorage::F16(
            flat.call_method1("astype", ("float16",))?
                .call_method0("tolist")?
                .extract::<Vec<f32>>()?
                .into_iter()
                .map(f16::from_f32)
                .collect(),
        )),
        DType::F32 => Ok(ArrayStorage::F32(
            flat.call_method1("astype", ("float32",))?
                .call_method0("tolist")?
                .extract::<Vec<f32>>()?,
        )),
        DType::F64 => Ok(ArrayStorage::F64(
            flat.call_method1("astype", ("float64",))?
                .call_method0("tolist")?
                .extract::<Vec<f64>>()?,
        )),
        DType::Complex64 => {
            let complex = flat.call_method1("astype", ("complex64",))?;
            let real = complex
                .getattr("real")?
                .call_method0("tolist")?
                .extract::<Vec<f32>>()?;
            let imag = complex
                .getattr("imag")?
                .call_method0("tolist")?
                .extract::<Vec<f32>>()?;
            Ok(ArrayStorage::Complex64(
                real.into_iter().zip(imag).collect(),
            ))
        }
        DType::Complex128 => {
            let complex = flat.call_method1("astype", ("complex128",))?;
            let real = complex
                .getattr("real")?
                .call_method0("tolist")?
                .extract::<Vec<f64>>()?;
            let imag = complex
                .getattr("imag")?
                .call_method0("tolist")?
                .extract::<Vec<f64>>()?;
            Ok(ArrayStorage::Complex128(
                real.into_iter().zip(imag).collect(),
            ))
        }
        DType::Str => Ok(ArrayStorage::String(
            flat.call_method0("tolist")?.extract::<Vec<String>>()?,
        )),
        unsupported => Err(PyTypeError::new_err(format!(
            "{context}: unsupported structured field dtype {}",
            unsupported.name()
        ))),
    }
}

#[allow(dead_code)]
fn extract_structured_leaf_columns(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    base_ndim: usize,
    context: &str,
) -> PyResult<Vec<(DType, ArrayStorage, usize)>> {
    let numpy = py.import("numpy")?;
    let array = numpy.call_method1("asarray", (value,))?;
    let dtype = array.getattr("dtype")?;
    let names = dtype.getattr("names")?;

    if names.is_none() {
        let dtype_name = dtype.getattr("name")?.extract::<String>()?;
        let parsed_dtype = DType::parse(&dtype_name).ok_or_else(|| {
            PyTypeError::new_err(format!(
                "{context}: unsupported structured field dtype {dtype_name}"
            ))
        })?;
        let shape = array.getattr("shape")?.extract::<Vec<usize>>()?;
        let width = shape
            .iter()
            .skip(base_ndim)
            .copied()
            .product::<usize>()
            .max(1);
        let flat = array.call_method1("reshape", (-1,))?;
        let storage = extract_storage_from_flat_array(&flat, parsed_dtype, context)?;
        return Ok(vec![(parsed_dtype, storage, width)]);
    }

    let field_names = names.extract::<Vec<String>>()?;
    let mut columns = Vec::new();
    for field_name in field_names {
        let field_array = array.get_item(field_name.as_str())?;
        columns.extend(extract_structured_leaf_columns(
            py,
            &field_array,
            base_ndim,
            context,
        )?);
    }
    Ok(columns)
}

#[allow(dead_code)]
fn promote_structured_leaf_dtypes(dtypes: &[DType]) -> Option<DType> {
    let mut iter = dtypes.iter().copied();
    let first = iter.next()?;
    Some(iter.fold(first, promote))
}

#[allow(dead_code)]
fn build_numpy_array_from_interleaved_storage(
    py: Python<'_>,
    shape: &[usize],
    columns: &[(ArrayStorage, usize)],
    num_records: usize,
    target_dtype: DType,
) -> PyResult<Py<PyAny>> {
    let numpy = py.import("numpy")?;
    let total_width = columns.iter().map(|(_, width)| *width).sum::<usize>();
    let total_len = num_records.saturating_mul(total_width);
    let kwargs = PyDict::new(py);

    let array = match target_dtype {
        DType::Bool => {
            let mut values = Vec::with_capacity(total_len);
            for record in 0..num_records {
                for (column, width) in columns {
                    let ArrayStorage::Bool(column_values) = column else {
                        unreachable!("bool target must use bool storage");
                    };
                    let base = record * *width;
                    for component in 0..*width {
                        values.push(column_values[base + component]);
                    }
                }
            }
            kwargs.set_item("dtype", "bool_")?;
            numpy.call_method(
                "array",
                (PyList::new(py, values.iter().copied())?,),
                Some(&kwargs),
            )?
        }
        DType::I8 => {
            let mut values = Vec::with_capacity(total_len);
            for record in 0..num_records {
                for (column, width) in columns {
                    let ArrayStorage::I8(column_values) = column else {
                        unreachable!("int8 target must use int8 storage");
                    };
                    let base = record * *width;
                    for component in 0..*width {
                        values.push(column_values[base + component]);
                    }
                }
            }
            kwargs.set_item("dtype", "int8")?;
            numpy.call_method(
                "array",
                (PyList::new(py, values.iter().copied())?,),
                Some(&kwargs),
            )?
        }
        DType::I16 => {
            let mut values = Vec::with_capacity(total_len);
            for record in 0..num_records {
                for (column, width) in columns {
                    let ArrayStorage::I16(column_values) = column else {
                        unreachable!("int16 target must use int16 storage");
                    };
                    let base = record * *width;
                    for component in 0..*width {
                        values.push(column_values[base + component]);
                    }
                }
            }
            kwargs.set_item("dtype", "int16")?;
            numpy.call_method(
                "array",
                (PyList::new(py, values.iter().copied())?,),
                Some(&kwargs),
            )?
        }
        DType::I32 => {
            let mut values = Vec::with_capacity(total_len);
            for record in 0..num_records {
                for (column, width) in columns {
                    let ArrayStorage::I32(column_values) = column else {
                        unreachable!("int32 target must use int32 storage");
                    };
                    let base = record * *width;
                    for component in 0..*width {
                        values.push(column_values[base + component]);
                    }
                }
            }
            kwargs.set_item("dtype", "int32")?;
            numpy.call_method(
                "array",
                (PyList::new(py, values.iter().copied())?,),
                Some(&kwargs),
            )?
        }
        DType::I64 => {
            let mut values = Vec::with_capacity(total_len);
            for record in 0..num_records {
                for (column, width) in columns {
                    let ArrayStorage::I64(column_values) = column else {
                        unreachable!("int64 target must use int64 storage");
                    };
                    let base = record * *width;
                    for component in 0..*width {
                        values.push(column_values[base + component]);
                    }
                }
            }
            kwargs.set_item("dtype", "int64")?;
            numpy.call_method(
                "array",
                (PyList::new(py, values.iter().copied())?,),
                Some(&kwargs),
            )?
        }
        DType::U8 => {
            let mut values = Vec::with_capacity(total_len);
            for record in 0..num_records {
                for (column, width) in columns {
                    let ArrayStorage::U8(column_values) = column else {
                        unreachable!("uint8 target must use uint8 storage");
                    };
                    let base = record * *width;
                    for component in 0..*width {
                        values.push(column_values[base + component]);
                    }
                }
            }
            kwargs.set_item("dtype", "uint8")?;
            numpy.call_method(
                "array",
                (PyList::new(py, values.iter().copied())?,),
                Some(&kwargs),
            )?
        }
        DType::U16 => {
            let mut values = Vec::with_capacity(total_len);
            for record in 0..num_records {
                for (column, width) in columns {
                    let ArrayStorage::U16(column_values) = column else {
                        unreachable!("uint16 target must use uint16 storage");
                    };
                    let base = record * *width;
                    for component in 0..*width {
                        values.push(column_values[base + component]);
                    }
                }
            }
            kwargs.set_item("dtype", "uint16")?;
            numpy.call_method(
                "array",
                (PyList::new(py, values.iter().copied())?,),
                Some(&kwargs),
            )?
        }
        DType::U32 => {
            let mut values = Vec::with_capacity(total_len);
            for record in 0..num_records {
                for (column, width) in columns {
                    let ArrayStorage::U32(column_values) = column else {
                        unreachable!("uint32 target must use uint32 storage");
                    };
                    let base = record * *width;
                    for component in 0..*width {
                        values.push(column_values[base + component]);
                    }
                }
            }
            kwargs.set_item("dtype", "uint32")?;
            numpy.call_method(
                "array",
                (PyList::new(py, values.iter().copied())?,),
                Some(&kwargs),
            )?
        }
        DType::U64 => {
            let mut values = Vec::with_capacity(total_len);
            for record in 0..num_records {
                for (column, width) in columns {
                    let ArrayStorage::U64(column_values) = column else {
                        unreachable!("uint64 target must use uint64 storage");
                    };
                    let base = record * *width;
                    for component in 0..*width {
                        values.push(column_values[base + component]);
                    }
                }
            }
            kwargs.set_item("dtype", "uint64")?;
            numpy.call_method(
                "array",
                (PyList::new(py, values.iter().copied())?,),
                Some(&kwargs),
            )?
        }
        DType::F16 => {
            let mut values = Vec::with_capacity(total_len);
            for record in 0..num_records {
                for (column, width) in columns {
                    let ArrayStorage::F16(column_values) = column else {
                        unreachable!("float16 target must use float16 storage");
                    };
                    let base = record * *width;
                    for component in 0..*width {
                        values.push(f32::from(column_values[base + component]));
                    }
                }
            }
            kwargs.set_item("dtype", "float16")?;
            numpy.call_method(
                "array",
                (PyList::new(py, values.iter().copied())?,),
                Some(&kwargs),
            )?
        }
        DType::F32 => {
            let mut values = Vec::with_capacity(total_len);
            for record in 0..num_records {
                for (column, width) in columns {
                    let ArrayStorage::F32(column_values) = column else {
                        unreachable!("float32 target must use float32 storage");
                    };
                    let base = record * *width;
                    for component in 0..*width {
                        values.push(column_values[base + component]);
                    }
                }
            }
            kwargs.set_item("dtype", "float32")?;
            numpy.call_method(
                "array",
                (PyList::new(py, values.iter().copied())?,),
                Some(&kwargs),
            )?
        }
        DType::F64 => {
            let mut values = Vec::with_capacity(total_len);
            for record in 0..num_records {
                for (column, width) in columns {
                    let ArrayStorage::F64(column_values) = column else {
                        unreachable!("float64 target must use float64 storage");
                    };
                    let base = record * *width;
                    for component in 0..*width {
                        values.push(column_values[base + component]);
                    }
                }
            }
            kwargs.set_item("dtype", "float64")?;
            numpy.call_method(
                "array",
                (PyList::new(py, values.iter().copied())?,),
                Some(&kwargs),
            )?
        }
        DType::Complex64 => {
            let builtins = py.import("builtins")?;
            let mut values = Vec::with_capacity(total_len);
            for record in 0..num_records {
                for (column, width) in columns {
                    let ArrayStorage::Complex64(column_values) = column else {
                        unreachable!("complex64 target must use complex64 storage");
                    };
                    let base = record * *width;
                    for component in 0..*width {
                        let (re, im) = column_values[base + component];
                        values.push(builtins.getattr("complex")?.call1((re, im))?.unbind());
                    }
                }
            }
            kwargs.set_item("dtype", "complex64")?;
            numpy.call_method("array", (PyList::new(py, values.iter())?,), Some(&kwargs))?
        }
        DType::Complex128 => {
            let builtins = py.import("builtins")?;
            let mut values = Vec::with_capacity(total_len);
            for record in 0..num_records {
                for (column, width) in columns {
                    let ArrayStorage::Complex128(column_values) = column else {
                        unreachable!("complex128 target must use complex128 storage");
                    };
                    let base = record * *width;
                    for component in 0..*width {
                        let (re, im) = column_values[base + component];
                        values.push(builtins.getattr("complex")?.call1((re, im))?.unbind());
                    }
                }
            }
            kwargs.set_item("dtype", "complex128")?;
            numpy.call_method("array", (PyList::new(py, values.iter())?,), Some(&kwargs))?
        }
        DType::Str => {
            let mut values = Vec::with_capacity(total_len);
            for record in 0..num_records {
                for (column, width) in columns {
                    let ArrayStorage::String(column_values) = column else {
                        unreachable!("str target must use string storage");
                    };
                    let base = record * *width;
                    for component in 0..*width {
                        values.push(column_values[base + component].clone());
                    }
                }
            }
            kwargs.set_item("dtype", "str")?;
            numpy.call_method("array", (PyList::new(py, values.iter())?,), Some(&kwargs))?
        }
        unsupported => {
            return Err(PyTypeError::new_err(format!(
                "structured_to_unstructured: unsupported output dtype {}",
                unsupported.name()
            )));
        }
    };

    let mut output_shape = shape.to_vec();
    output_shape.push(total_width);
    let output_shape = PyTuple::new(py, output_shape.iter().copied())?;
    Ok(array.call_method1("reshape", (&output_shape,))?.unbind())
}

fn validate_cpu_device_kwarg(py: Python<'_>, device: Option<Py<PyAny>>) -> PyResult<()> {
    let Some(device) = device else {
        return Ok(());
    };

    let device = device.bind(py);
    if device.is_none() {
        return Ok(());
    }

    let device_text = device.str()?.extract::<String>()?;
    if device_text == "cpu" {
        return Ok(());
    }

    Err(PyValueError::new_err(format!(
        "Device not understood. Only \"cpu\" is allowed, but received: {device_text}",
    )))
}

#[pyfunction]
#[pyo3(signature = (arr, dtype=None, copy=false, casting="unsafe"))]
fn structured_to_unstructured(
    py: Python<'_>,
    arr: Py<PyAny>,
    dtype: Option<Py<PyAny>>,
    copy: bool,
    casting: &str,
) -> PyResult<Py<PyAny>> {
    // Delegate to numpy.lib.recfunctions so scalar records, nested subarrays,
    // casting rules, and dtype inference match NumPy exactly.
    let recfunctions = py.import("numpy.lib.recfunctions")?;
    let kwargs = PyDict::new(py);
    if let Some(dtype) = dtype {
        kwargs.set_item("dtype", dtype.bind(py))?;
    }
    kwargs.set_item("copy", copy)?;
    kwargs.set_item("casting", casting)?;
    Ok(recfunctions
        .getattr("structured_to_unstructured")?
        .call((arr.bind(py),), Some(&kwargs))?
        .unbind())
}

fn extract_ix_array(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    context: &str,
) -> PyResult<UFuncArray> {
    let numpy = py.import("numpy")?;
    let builtins = py.import("builtins")?;
    let ndarray = numpy.getattr("ndarray")?;
    let is_ndarray = builtins
        .call_method1("isinstance", (value, ndarray))?
        .extract::<bool>()?;

    let mut array = numpy.call_method1("asarray", (value,))?;
    if array.getattr("ndim")?.extract::<usize>()? != 1 {
        return Err(PyValueError::new_err("Cross index must be 1 dimensional"));
    }

    let dtype = array.getattr("dtype")?;
    let kind = dtype.getattr("kind")?.extract::<String>()?;
    if kind == "b" {
        array = array.call_method0("nonzero")?.get_item(0)?;
    } else if !is_ndarray && array.getattr("size")?.extract::<usize>()? == 0 {
        array = array.call_method1("astype", ("int64",))?;
    }

    extract_precise_numeric_array(py, &array, context)
}

fn validate_meshgrid_indexing(indexing: &str) -> PyResult<()> {
    match indexing {
        "xy" | "ij" => Ok(()),
        _ => Err(PyValueError::new_err(format!(
            "meshgrid: indexing must be 'xy' or 'ij', got '{indexing}'",
        ))),
    }
}

fn meshgrid_output_axis(index: usize, ndim: usize, indexing: &str) -> usize {
    if indexing == "xy" && ndim >= 2 {
        match index {
            0 => 1,
            1 => 0,
            dim => dim,
        }
    } else {
        index
    }
}

fn normalize_meshgrid_inputs(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
) -> PyResult<Vec<Py<PyAny>>> {
    let numpy = py.import("numpy")?;
    args.iter()
        .map(|value| {
            let array = numpy.call_method1("asarray", (value,))?;
            Ok(array.call_method1("reshape", (-1,))?.unbind())
        })
        .collect()
}

fn meshgrid_rust_compatible(py: Python<'_>, arrays: &[Py<PyAny>]) -> PyResult<bool> {
    for array in arrays {
        let kind = array
            .bind(py)
            .getattr("dtype")?
            .getattr("kind")?
            .extract::<String>()?;
        if !matches!(kind.as_str(), "b" | "i" | "u" | "f") {
            return Ok(false);
        }
    }
    Ok(true)
}

fn parse_grid_slice(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    context: &str,
) -> PyResult<ParsedGridAxis> {
    let start = if value.getattr("start")?.is_none() {
        0_i64.into_pyobject(py)?.into_any().unbind()
    } else {
        value.getattr("start")?.unbind()
    };
    let stop = if value.getattr("stop")?.is_none() {
        0_i64.into_pyobject(py)?.into_any().unbind()
    } else {
        value.getattr("stop")?.unbind()
    };
    let step_obj = if value.getattr("step")?.is_none() {
        1_i64.into_pyobject(py)?.into_any().unbind()
    } else {
        value.getattr("step")?.unbind()
    };

    let numpy = py.import("numpy")?;
    let builtins = py.import("builtins")?;
    let is_complex = numpy
        .getattr("iscomplexobj")?
        .call1((step_obj.bind(py),))?
        .extract::<bool>()?;

    let start_f64 = start.bind(py).extract::<f64>().map_err(|_| {
        PyTypeError::new_err(format!(
            "{context}: slice start must be an int or float-like scalar",
        ))
    })?;
    let stop_f64 = stop.bind(py).extract::<f64>().map_err(|_| {
        PyTypeError::new_err(format!(
            "{context}: slice stop must be an int or float-like scalar",
        ))
    })?;

    let (spec, step_for_dtype) = if is_complex {
        let magnitude = builtins.call_method1("abs", (step_obj.bind(py),))?;
        let num = magnitude.extract::<f64>().map_err(|_| {
            PyTypeError::new_err(format!(
                "{context}: complex slice step must have a numeric magnitude",
            ))
        })?;
        (
            GridSpec::Linspace {
                start: start_f64,
                stop: stop_f64,
                num: num as usize,
            },
            magnitude.unbind(),
        )
    } else {
        let step_f64 = step_obj.bind(py).extract::<f64>().map_err(|_| {
            PyTypeError::new_err(format!(
                "{context}: slice step must be an int or float-like scalar",
            ))
        })?;
        (
            GridSpec::Arange {
                start: start_f64,
                stop: stop_f64,
                step: step_f64,
            },
            step_obj.clone_ref(py),
        )
    };

    Ok(ParsedGridAxis {
        spec,
        start,
        stop,
        step_for_dtype,
    })
}

fn parse_grid_key(
    py: Python<'_>,
    key: &Bound<'_, PyAny>,
    context: &str,
) -> PyResult<(Vec<GridSpec>, Py<PyAny>, bool)> {
    let numpy = py.import("numpy")?;

    if let Ok(tuple) = key.downcast::<PyTuple>() {
        let mut specs = Vec::with_capacity(tuple.len());
        let mut dtype_args = vec![0_i64.into_pyobject(py)?.into_any().unbind()];
        for (index, item) in tuple.try_iter()?.enumerate() {
            let item = item?;
            let slice = item.downcast::<pyo3::types::PySlice>().map_err(|_| {
                PyTypeError::new_err(format!(
                    "{context}: index {index} must be a slice expression",
                ))
            })?;
            let parsed = parse_grid_slice(py, slice.as_any(), context)?;
            specs.push(parsed.spec);
            dtype_args.push(parsed.start);
            dtype_args.push(parsed.stop);
            dtype_args.push(parsed.step_for_dtype);
        }
        let dtype = numpy.getattr("result_type")?.call1(PyTuple::new(
            py,
            dtype_args.iter().map(|value| value.bind(py)),
        )?)?;
        Ok((specs, dtype.unbind(), true))
    } else {
        let slice = key.downcast::<pyo3::types::PySlice>().map_err(|_| {
            PyTypeError::new_err(format!("{context}: expected slice syntax, e.g. [0:5:2]"))
        })?;
        let parsed = parse_grid_slice(py, slice.as_any(), context)?;
        let dtype = match parsed.spec {
            GridSpec::Linspace { .. } => numpy.getattr("result_type")?.call1((
                parsed.start.bind(py),
                parsed.stop.bind(py),
                parsed.step_for_dtype.bind(py),
            ))?,
            GridSpec::Arange { .. } => numpy
                .getattr("arange")?
                .call1((
                    parsed.start.bind(py),
                    parsed.stop.bind(py),
                    parsed.step_for_dtype.bind(py),
                ))?
                .getattr("dtype")?,
        };
        Ok((vec![parsed.spec], dtype.unbind(), false))
    }
}

fn cast_numpy_array_dtype(
    py: Python<'_>,
    array: Py<PyAny>,
    dtype: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    Ok(array
        .bind(py)
        .call_method(
            "astype",
            (dtype,),
            Some(&{
                let kwargs = PyDict::new(py);
                kwargs.set_item("copy", false)?;
                kwargs
            }),
        )?
        .unbind())
}

fn build_grid_numpy_outputs(
    py: Python<'_>,
    arrays: &[UFuncArray],
    dtype: &Bound<'_, PyAny>,
    sparse: bool,
    tuple_input: bool,
) -> PyResult<Py<PyAny>> {
    if sparse {
        if tuple_input {
            let outputs = arrays
                .iter()
                .map(|array| {
                    let output = build_numpy_array_from_ufunc(py, array)?;
                    cast_numpy_array_dtype(py, output, dtype)
                })
                .collect::<PyResult<Vec<_>>>()?;
            build_numpy_tuple_from_pyarrays(py, &outputs)
        } else {
            let output = build_numpy_array_from_ufunc(py, &arrays[0])?;
            cast_numpy_array_dtype(py, output, dtype)
        }
    } else if tuple_input {
        let numpy = py.import("numpy")?;
        let tuple = build_numpy_tuple_from_ufuncs(py, arrays)?;
        let stacked = numpy.getattr("stack")?.call1((tuple.bind(py),))?;
        cast_numpy_array_dtype(py, stacked.unbind(), dtype)
    } else {
        let output = build_numpy_array_from_ufunc(py, &arrays[0])?;
        cast_numpy_array_dtype(py, output, dtype)
    }
}

fn grid_getitem(py: Python<'_>, key: &Bound<'_, PyAny>, sparse: bool) -> PyResult<Py<PyAny>> {
    let context = if sparse { "ogrid" } else { "mgrid" };
    let (specs, dtype, tuple_input) = parse_grid_key(py, key, context)?;
    let arrays = if sparse {
        UFuncArray::ogrid_spec(&specs)
    } else {
        UFuncArray::mgrid_spec(&specs)
    };
    build_grid_numpy_outputs(py, &arrays, dtype.bind(py), sparse, tuple_input)
}

fn axis_concatenator_items(key: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
    if let Ok(tuple) = key.downcast::<PyTuple>() {
        tuple.try_iter()?.map(|item| Ok(item?.unbind())).collect()
    } else {
        Ok(vec![key.clone().unbind()])
    }
}

fn axis_concatenator_slice_array(
    py: Python<'_>,
    kind: AxisConcatenatorKind,
    value: &Bound<'_, PyAny>,
) -> PyResult<UFuncArray> {
    let numpy = py.import("numpy")?;
    let array = numpy
        .getattr(kind.context())?
        .call_method1("__getitem__", (value,))?;
    extract_precise_numeric_array(py, &array, kind.context())
}

fn axis_concatenator_array(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    kind: AxisConcatenatorKind,
) -> PyResult<Option<UFuncArray>> {
    if value.extract::<String>().is_ok() {
        return Ok(None);
    }

    if let Ok(slice) = value.downcast::<pyo3::types::PySlice>() {
        return axis_concatenator_slice_array(py, kind, slice.as_any()).map(Some);
    }

    match extract_precise_numeric_array(py, value, kind.context()) {
        Ok(array) => Ok(Some(array)),
        Err(_) => Ok(None),
    }
}

fn axis_concatenator_numpy_fallback(
    py: Python<'_>,
    kind: AxisConcatenatorKind,
    key: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let numpy = py.import("numpy")?;
    Ok(numpy
        .getattr(kind.context())?
        .call_method1("__getitem__", (key,))?
        .unbind())
}

fn axis_concatenator_getitem(
    py: Python<'_>,
    key: &Bound<'_, PyAny>,
    kind: AxisConcatenatorKind,
) -> PyResult<Py<PyAny>> {
    let arrays = axis_concatenator_items(key)?
        .into_iter()
        .map(|item| axis_concatenator_array(py, item.bind(py), kind))
        .collect::<PyResult<Vec<_>>>()?;

    if arrays.iter().any(Option::is_none) {
        return axis_concatenator_numpy_fallback(py, kind, key);
    }

    let materialized = arrays.into_iter().flatten().collect::<Vec<_>>();
    let result = kind.concatenate(&materialized).map_err(map_ufunc_error)?;
    build_numpy_array_from_ufunc(py, &result)
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
        ArrayStorage::F16(values) => (
            PyList::new(py, values.iter().map(|value| f32::from(*value)))?.into_any(),
            "float16",
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

fn build_numpy_complex_array_from_interleaved(
    py: Python<'_>,
    array: &UFuncArray,
) -> PyResult<Py<PyAny>> {
    let Some((&2, logical_shape)) = array.shape().split_last() else {
        return Err(PyTypeError::new_err(
            "complex output must use an interleaved trailing dimension of size 2",
        ));
    };

    let numpy = py.import("numpy")?;
    let builtins = py.import("builtins")?;
    let complex_values = array
        .values()
        .chunks_exact(2)
        .map(|chunk| builtins.getattr("complex")?.call1((chunk[0], chunk[1])))
        .collect::<PyResult<Vec<_>>>()?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", numpy.getattr("complex128")?)?;
    let result = numpy
        .getattr("array")?
        .call((PyList::new(py, complex_values.iter())?,), Some(&kwargs))?;
    if logical_shape.len() <= 1 {
        Ok(result.unbind())
    } else {
        Ok(result
            .call_method1(
                "reshape",
                (PyTuple::new(py, logical_shape.iter().copied())?,),
            )?
            .unbind())
    }
}

fn build_numpy_eigvals_vector_from_flat_interleaved(
    py: Python<'_>,
    values: &[f64],
    input_dtype: DType,
) -> PyResult<Py<PyAny>> {
    if !values.len().is_multiple_of(2) {
        return Err(PyTypeError::new_err(
            "complex output must contain flat interleaved real/imag pairs",
        ));
    }

    let numpy = py.import("numpy")?;
    let all_real = values.chunks_exact(2).all(|chunk| chunk[1] == 0.0);
    if all_real {
        let kwargs = PyDict::new(py);
        match input_dtype {
            DType::F32 => {
                let real_values = values
                    .chunks_exact(2)
                    .map(|chunk| chunk[0] as f32)
                    .collect::<Vec<_>>();
                kwargs.set_item("dtype", numpy.getattr("float32")?)?;
                return Ok(numpy
                    .getattr("array")?
                    .call(
                        (PyList::new(py, real_values.iter().copied())?,),
                        Some(&kwargs),
                    )?
                    .unbind());
            }
            _ => {
                let real_values = values
                    .chunks_exact(2)
                    .map(|chunk| chunk[0])
                    .collect::<Vec<_>>();
                kwargs.set_item("dtype", numpy.getattr("float64")?)?;
                return Ok(numpy
                    .getattr("array")?
                    .call(
                        (PyList::new(py, real_values.iter().copied())?,),
                        Some(&kwargs),
                    )?
                    .unbind());
            }
        }
    }

    let builtins = py.import("builtins")?;
    let complex_values = values
        .chunks_exact(2)
        .map(|chunk| builtins.getattr("complex")?.call1((chunk[0], chunk[1])))
        .collect::<PyResult<Vec<_>>>()?;
    let kwargs = PyDict::new(py);
    let complex_dtype = match input_dtype {
        DType::F32 => numpy.getattr("complex64")?,
        _ => numpy.getattr("complex128")?,
    };
    kwargs.set_item("dtype", complex_dtype)?;
    Ok(numpy
        .getattr("array")?
        .call((PyList::new(py, complex_values.iter())?,), Some(&kwargs))?
        .unbind())
}

#[allow(dead_code)]
fn extract_complex_interleaved_array(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    context: &str,
) -> PyResult<UFuncArray> {
    let numpy = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", numpy.getattr("complex128")?)?;
    let array = numpy.call_method("asarray", (value,), Some(&kwargs))?;
    let shape = array.getattr("shape")?.extract::<Vec<usize>>()?;
    if shape.len() != 1 {
        return Err(PyTypeError::new_err(format!(
            "{context}: expected a 1-D complex array"
        )));
    }
    let flat = array.call_method1("reshape", (-1,))?;
    let real = flat
        .getattr("real")?
        .call_method0("tolist")?
        .extract::<Vec<f64>>()?;
    let imag = flat
        .getattr("imag")?
        .call_method0("tolist")?
        .extract::<Vec<f64>>()?;
    let mut values = Vec::with_capacity(real.len() * 2);
    for (&re, &im) in real.iter().zip(&imag) {
        values.push(re);
        values.push(im);
    }
    UFuncArray::new(vec![shape[0], 2], values, DType::F64).map_err(map_ufunc_error)
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

fn build_numpy_list_from_ufuncs(py: Python<'_>, arrays: &[UFuncArray]) -> PyResult<Py<PyAny>> {
    let arrays = arrays
        .iter()
        .map(|array| build_numpy_array_from_ufunc(py, array))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, arrays.iter().map(|array| array.bind(py)))?
        .into_any()
        .unbind())
}

fn build_numpy_scalar_or_array_from_ufunc(
    py: Python<'_>,
    array: &UFuncArray,
) -> PyResult<Py<PyAny>> {
    let output = build_numpy_array_from_ufunc(py, array)?;
    if array.shape().is_empty() {
        Ok(output.bind(py).get_item(())?.unbind())
    } else {
        Ok(output)
    }
}

fn build_numpy_index_tuple_from_ufuncs(
    py: Python<'_>,
    arrays: &[UFuncArray],
    scalar_output: bool,
) -> PyResult<Py<PyAny>> {
    if !scalar_output {
        return build_numpy_tuple_from_ufuncs(py, arrays);
    }

    let scalars = arrays
        .iter()
        .map(|array| {
            let output = build_numpy_array_from_ufunc(py, array)?;
            output.bind(py).get_item(())
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, scalars.iter())?.into_any().unbind())
}

fn build_numpy_tuple_from_pyarrays(py: Python<'_>, arrays: &[Py<PyAny>]) -> PyResult<Py<PyAny>> {
    Ok(PyTuple::new(py, arrays.iter().map(|array| array.bind(py)))?
        .into_any()
        .unbind())
}

fn build_meshgrid_numpy_outputs(
    py: Python<'_>,
    arrays: &[Py<PyAny>],
    indexing: &str,
    sparse: bool,
    copy: bool,
) -> PyResult<Py<PyAny>> {
    validate_meshgrid_indexing(indexing)?;
    if arrays.is_empty() {
        return build_numpy_tuple_from_pyarrays(py, &[]);
    }

    let numpy = py.import("numpy")?;
    let ndim = arrays.len();
    let reshaped = arrays
        .iter()
        .enumerate()
        .map(|(index, array)| {
            let axis = meshgrid_output_axis(index, ndim, indexing);
            let mut shape = vec![1_usize; ndim];
            shape[axis] = array.bind(py).len()?;
            let shape = PyTuple::new(py, shape.iter().copied())?;
            Ok(array.bind(py).call_method1("reshape", (&shape,))?.unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;

    let outputs = if sparse {
        reshaped
    } else {
        let broadcast = numpy.getattr("broadcast_arrays")?.call1(PyTuple::new(
            py,
            reshaped.iter().map(|array| array.bind(py)),
        )?)?;
        let broadcast = broadcast.downcast::<PyTuple>()?;
        broadcast
            .try_iter()?
            .map(|item| Ok(item?.unbind()))
            .collect::<PyResult<Vec<_>>>()?
    };

    let outputs = if copy {
        outputs
            .into_iter()
            .map(|array| Ok(array.bind(py).call_method0("copy")?.unbind()))
            .collect::<PyResult<Vec<_>>>()?
    } else {
        outputs
    };

    build_numpy_tuple_from_pyarrays(py, &outputs)
}

fn frompyfunc_display_name(callable: &Bound<'_, PyAny>) -> String {
    let base_name = callable
        .getattr("__name__")
        .and_then(|name| name.extract::<String>())
        .unwrap_or_else(|_| "pyfunc".to_string());
    format!("{base_name} (vectorized)")
}

fn parse_frompyfunc_identity(
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<FromPyFuncReduceIdentity<Py<PyAny>>> {
    let Some(kwargs) = kwargs else {
        return Ok(FromPyFuncReduceIdentity::Omitted);
    };

    let mut identity = FromPyFuncReduceIdentity::Omitted;
    for (key, value) in kwargs.iter() {
        let key = key
            .extract::<String>()
            .map_err(|_| PyTypeError::new_err("frompyfunc() keywords must be strings"))?;
        match key.as_str() {
            "identity" => {
                identity = if value.is_none() {
                    FromPyFuncReduceIdentity::ReorderableNone
                } else {
                    FromPyFuncReduceIdentity::Value(value.unbind())
                };
            }
            other => {
                return Err(PyTypeError::new_err(format!(
                    "frompyfunc() got an unexpected keyword argument '{other}'",
                )));
            }
        }
    }

    Ok(identity)
}

fn extract_object_array_input(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    context: &str,
) -> PyResult<(Vec<usize>, Vec<Py<PyAny>>)> {
    let numpy = py.import("numpy")?;
    let builtins = py.import("builtins")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", builtins.getattr("object")?)?;
    let array = numpy.call_method("asarray", (value,), Some(&kwargs))?;
    let shape = array.getattr("shape")?.extract::<Vec<usize>>()?;
    let flat = array.call_method1("reshape", (-1,))?;
    let values = flat
        .call_method0("tolist")?
        .extract::<Vec<Py<PyAny>>>()
        .map_err(|_| {
            PyTypeError::new_err(format!("{context}: failed to normalize object array"))
        })?;
    Ok((shape, values))
}

fn build_numpy_object_array_from_flat_values(
    py: Python<'_>,
    shape: &[usize],
    values: &[Py<PyAny>],
) -> PyResult<Py<PyAny>> {
    let numpy = py.import("numpy")?;
    let builtins = py.import("builtins")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", builtins.getattr("object")?)?;
    let list = PyList::new(py, values.iter().map(|value| value.bind(py)))?;
    let array = numpy.call_method("array", (list,), Some(&kwargs))?;
    let shape = PyTuple::new(py, shape.iter().copied())?;
    Ok(array.call_method1("reshape", (shape,))?.unbind())
}

fn build_numpy_scalar_or_array_from_object_values(
    py: Python<'_>,
    shape: &[usize],
    values: &[Py<PyAny>],
) -> PyResult<Py<PyAny>> {
    let array = build_numpy_object_array_from_flat_values(py, shape, values)?;
    if shape.is_empty() {
        Ok(array.bind(py).get_item(())?.unbind())
    } else {
        Ok(array)
    }
}

fn normalize_reduce_out_argument(py: Python<'_>, out: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let candidate = if let Ok(tuple) = out.downcast::<PyTuple>() {
        if tuple.len() != 1 {
            return Err(PyTypeError::new_err("output must be an array"));
        }
        tuple.get_item(0)?
    } else if out.downcast::<PyList>().is_ok() {
        return Err(PyTypeError::new_err("output must be an array"));
    } else {
        out.clone()
    };

    if require_numpy_ndarray(py, &candidate, "reduce(out)").is_err() {
        return Err(PyTypeError::new_err("output must be an array"));
    }
    Ok(candidate.unbind())
}

fn extract_frompyfunc_reduce_axis_spec(
    py: Python<'_>,
    axis: Option<Py<PyAny>>,
) -> PyResult<FromPyFuncReduceAxisSpec> {
    let Some(axis) = axis else {
        return Ok(FromPyFuncReduceAxisSpec::Default);
    };

    let axis = axis.bind(py);
    if axis.is_none() {
        return Ok(FromPyFuncReduceAxisSpec::All);
    }

    if let Ok(axis) = axis.extract::<isize>() {
        return Ok(FromPyFuncReduceAxisSpec::Axis(axis));
    }

    if let Ok(iter) = axis.try_iter() {
        let axes = iter
            .enumerate()
            .map(|(index, item)| {
                item?.extract::<isize>().map_err(|_| {
                    PyTypeError::new_err(format!("reduce: axis[{index}] must be an integer"))
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(FromPyFuncReduceAxisSpec::Axes(axes));
    }

    Err(PyTypeError::new_err(
        "reduce: axis must be an integer, tuple/list of integers, or None",
    ))
}

fn extract_frompyfunc_where_mask(
    py: Python<'_>,
    shape: &[usize],
    where_value: &Bound<'_, PyAny>,
) -> PyResult<Option<Vec<bool>>> {
    if where_value.is_instance_of::<PyBool>() {
        return if where_value.extract::<bool>()? {
            Ok(None)
        } else {
            Ok(Some(vec![
                false;
                element_count(shape).map_err(|err| {
                    PyValueError::new_err(err.to_string())
                })?
            ]))
        };
    }

    let numpy = py.import("numpy")?;
    let builtins = py.import("builtins")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", builtins.getattr("bool")?)?;
    let mask = numpy.call_method("asarray", (where_value,), Some(&kwargs))?;
    let shape = PyTuple::new(py, shape.iter().copied())?;
    let broadcast = numpy.call_method1("broadcast_to", (mask, shape))?;
    let flat = broadcast.call_method1("reshape", (-1,))?;
    Ok(Some(flat.call_method0("tolist")?.extract::<Vec<bool>>()?))
}

struct ParsedFromPyFuncReduce {
    array: Py<PyAny>,
    axis_spec: FromPyFuncReduceAxisSpec,
    out: Option<Py<PyAny>>,
    keepdims: bool,
    initial: Option<Py<PyAny>>,
    where_arg: Option<Py<PyAny>>,
}

fn set_reduce_kwarg<T>(slot: &mut Option<T>, value: T, name: &str) -> PyResult<()> {
    if slot.is_some() {
        Err(PyTypeError::new_err(format!(
            "reduce() got multiple values for argument '{name}'",
        )))
    } else {
        *slot = Some(value);
        Ok(())
    }
}

fn parse_frompyfunc_reduce_call(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<ParsedFromPyFuncReduce> {
    if args.is_empty() {
        return Err(PyTypeError::new_err(
            "reduce() missing required argument 'array' (pos 1)",
        ));
    }
    if args.len() > 7 {
        return Err(PyTypeError::new_err(format!(
            "reduce() takes at most 7 arguments ({} given)",
            args.len()
        )));
    }

    let array = args.get_item(0)?.unbind();
    let mut axis = args.get_item(1).ok().map(|value| value.unbind());
    let mut out = args
        .get_item(3)
        .ok()
        .filter(|value| !value.is_none())
        .map(|value| value.unbind());
    let mut keepdims = args
        .get_item(4)
        .ok()
        .map(|value| value.extract::<bool>())
        .transpose()?
        .unwrap_or(false);
    let mut initial = args
        .get_item(5)
        .ok()
        .filter(|value| !value.is_none())
        .map(|value| value.unbind());
    let mut where_arg = args.get_item(6).ok().map(|value| value.unbind());

    if let Some(kwargs) = kwargs {
        for (key, value) in kwargs.iter() {
            let key = key
                .extract::<String>()
                .map_err(|_| PyTypeError::new_err("reduce() keywords must be strings"))?;
            match key.as_str() {
                "axis" => set_reduce_kwarg(&mut axis, value.unbind(), "axis")?,
                "dtype" => {}
                "out" => {
                    if !value.is_none() {
                        set_reduce_kwarg(&mut out, value.unbind(), "out")?;
                    }
                }
                "keepdims" => {
                    if args.get_item(4).is_ok() {
                        return Err(PyTypeError::new_err(
                            "reduce() got multiple values for argument 'keepdims'",
                        ));
                    }
                    keepdims = value.extract::<bool>()?;
                }
                "initial" => {
                    if !value.is_none() {
                        set_reduce_kwarg(&mut initial, value.unbind(), "initial")?;
                    }
                }
                "where" => set_reduce_kwarg(&mut where_arg, value.unbind(), "where")?,
                other => {
                    return Err(PyTypeError::new_err(format!(
                        "reduce() got an unexpected keyword argument '{other}'",
                    )));
                }
            }
        }
    }

    Ok(ParsedFromPyFuncReduce {
        array,
        axis_spec: extract_frompyfunc_reduce_axis_spec(py, axis)?,
        out,
        keepdims,
        initial,
        where_arg,
    })
}

impl PyFromPyFunc {
    fn new_checked(
        callable: Py<PyAny>,
        nin: usize,
        nout: usize,
        kwargs: Option<&Bound<'_, PyDict>>,
        py: Python<'_>,
    ) -> PyResult<Self> {
        if !callable.bind(py).is_callable() {
            return Err(PyTypeError::new_err(
                "frompyfunc: callable_obj must be callable",
            ));
        }
        let display_name = frompyfunc_display_name(callable.bind(py));
        let identity = parse_frompyfunc_identity(kwargs)?;

        Ok(Self {
            callable,
            display_name,
            identity,
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

            if self.nout == 0 {
                continue;
            }

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
impl PyMGridClass {
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        grid_getitem(py, key, false)
    }

    fn __repr__(&self) -> &str {
        "MGridClass()"
    }
}

#[pymethods]
impl PyOGridClass {
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        grid_getitem(py, key, true)
    }

    fn __repr__(&self) -> &str {
        "OGridClass()"
    }
}

#[pymethods]
impl PyRClass {
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        axis_concatenator_getitem(py, key, AxisConcatenatorKind::Rows)
    }

    fn __len__(&self) -> usize {
        0
    }

    fn __repr__(&self) -> &str {
        "RClass()"
    }
}

#[pymethods]
impl PyCClass {
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        axis_concatenator_getitem(py, key, AxisConcatenatorKind::Columns)
    }

    fn __len__(&self) -> usize {
        0
    }

    fn __repr__(&self) -> &str {
        "CClass()"
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
    #[pyo3(signature = (callable_obj, nin, nout, **kwargs))]
    fn new(
        callable_obj: Py<PyAny>,
        nin: usize,
        nout: usize,
        kwargs: Option<&Bound<'_, PyDict>>,
        py: Python<'_>,
    ) -> PyResult<Self> {
        Self::new_checked(callable_obj, nin, nout, kwargs, py)
    }

    #[getter]
    fn nin(&self) -> usize {
        self.nin
    }

    #[getter]
    fn nout(&self) -> usize {
        self.nout
    }

    #[getter]
    fn identity(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(match &self.identity {
            FromPyFuncReduceIdentity::Value(value) => value.clone_ref(py),
            FromPyFuncReduceIdentity::Omitted | FromPyFuncReduceIdentity::ReorderableNone => {
                py.None()
            }
        })
    }

    fn __call__(&self, py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
        self.call_bound(py, args)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn reduce(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        if self.nin != 2 {
            return Err(PyValueError::new_err(
                "reduce only supported for binary functions",
            ));
        }
        if self.nout != 1 {
            return Err(PyValueError::new_err(
                "reduce only supported for functions returning a single value",
            ));
        }

        let parsed = parse_frompyfunc_reduce_call(py, args, kwargs)?;
        let (shape, values) =
            extract_object_array_input(py, parsed.array.bind(py), "reduce(array)")?;
        let where_mask = match parsed.where_arg.as_ref() {
            Some(where_arg) => extract_frompyfunc_where_mask(py, &shape, where_arg.bind(py))?,
            None => None,
        };

        let (out_shape, out_values) = reduce_frompyfunc_values(
            &shape,
            &values,
            FromPyFuncReduceOptions {
                axis_spec: parsed.axis_spec,
                keepdims: parsed.keepdims,
                initial: parsed.initial,
                where_mask: where_mask.as_deref(),
                identity: &self.identity,
                op_name: &self.display_name,
            },
            |value| value.clone_ref(py),
            |lhs, rhs| {
                let call_args = PyTuple::new(py, [lhs.bind(py), rhs.bind(py)])?;
                Ok(self.callable.bind(py).call1(call_args)?.unbind())
            },
        )
        .map_err(|err| match err {
            FromPyFuncReduceError::UFunc(err) => PyValueError::new_err(err.to_string()),
            FromPyFuncReduceError::Callback(err) => err,
        })?;

        if let Some(out) = parsed.out {
            let out = normalize_reduce_out_argument(py, out.bind(py))?;
            let result = build_numpy_object_array_from_flat_values(py, &out_shape, &out_values)?;
            py.import("numpy")?
                .call_method1("copyto", (out.bind(py), result.bind(py)))?;
            Ok(out)
        } else {
            build_numpy_scalar_or_array_from_object_values(py, &out_shape, &out_values)
        }
    }

    fn __repr__(&self) -> String {
        format!("<ufunc '{}'>", self.display_name)
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
#[pyo3(signature = (callable_obj, nin, nout, **kwargs))]
fn frompyfunc(
    py: Python<'_>,
    callable_obj: Py<PyAny>,
    nin: usize,
    nout: usize,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyFromPyFunc> {
    PyFromPyFunc::new_checked(callable_obj, nin, nout, kwargs, py)
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
#[pyo3(signature = (x, weights=None, minlength=0))]
fn bincount(
    py: Python<'_>,
    x: Py<PyAny>,
    weights: Option<Py<PyAny>>,
    minlength: i64,
) -> PyResult<Py<PyAny>> {
    if minlength < 0 {
        return Err(PyValueError::new_err("'minlength' must not be negative"));
    }
    let x = extract_numeric_array(py, x.bind(py), "bincount(x)")?;
    let weights = weights
        .map(|w| extract_numeric_array(py, w.bind(py), "bincount(weights)"))
        .transpose()?;
    let result = x
        .bincount_with(weights.as_ref(), minlength as usize)
        .map_err(map_ufunc_error)?;
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

fn trapezoid_impl(
    py: Python<'_>,
    name: &str,
    y: Py<PyAny>,
    x: Option<Py<PyAny>>,
    dx: f64,
    axis: isize,
) -> PyResult<Py<PyAny>> {
    let y = extract_numeric_array(py, y.bind(py), &format!("{name}(y)"))?;
    let result = match x {
        Some(x) => {
            let x = extract_numeric_array(py, x.bind(py), &format!("{name}(x)"))?;
            y.trapezoid_x(&x, Some(axis))
        }
        None => y.trapezoid(dx, Some(axis)),
    }
    .map_err(map_ufunc_error)?;
    build_numpy_scalar_or_array_from_ufunc(py, &result)
}

#[pyfunction]
#[pyo3(signature = (y, x=None, dx=1.0, axis=-1))]
fn trapezoid(
    py: Python<'_>,
    y: Py<PyAny>,
    x: Option<Py<PyAny>>,
    dx: f64,
    axis: isize,
) -> PyResult<Py<PyAny>> {
    trapezoid_impl(py, "trapezoid", y, x, dx, axis)
}

#[pyfunction]
#[pyo3(signature = (y, x=None, dx=1.0, axis=-1))]
fn trapz(
    py: Python<'_>,
    y: Py<PyAny>,
    x: Option<Py<PyAny>>,
    dx: f64,
    axis: isize,
) -> PyResult<Py<PyAny>> {
    trapezoid_impl(py, "trapz", y, x, dx, axis)
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
fn flatnonzero(py: Python<'_>, a: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let a = extract_numeric_array(py, a.bind(py), "flatnonzero(a)")?;
    let result = a.flatnonzero();
    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction]
fn argwhere(py: Python<'_>, a: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let a = extract_numeric_array(py, a.bind(py), "argwhere(a)")?;
    let result = a.argwhere();
    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction]
#[pyo3(signature = (a, indices, axis=None))]
fn take(
    py: Python<'_>,
    a: Py<PyAny>,
    indices: Py<PyAny>,
    axis: Option<isize>,
) -> PyResult<Py<PyAny>> {
    let a = extract_numeric_array(py, a.bind(py), "take(a)")?;
    let (indices_shape, flat_indices) =
        extract_take_indices(py, indices.bind(py), "take(indices)")?;
    let mut result = a.take(&flat_indices, axis).map_err(map_ufunc_error)?;

    let desired_shape = match axis {
        None => indices_shape.clone(),
        Some(axis) => {
            let ndim = a.shape().len();
            let normalized_axis = if axis < 0 {
                axis + isize::try_from(ndim).map_err(|_| {
                    PyValueError::new_err("take: array rank exceeds signed axis range")
                })?
            } else {
                axis
            };

            if normalized_axis < 0
                || normalized_axis
                    >= isize::try_from(ndim).map_err(|_| {
                        PyValueError::new_err("take: array rank exceeds signed axis range")
                    })?
            {
                return Err(PyValueError::new_err(format!(
                    "take: axis {axis} is out of bounds for array of dimension {ndim}",
                )));
            }

            let normalized_axis = usize::try_from(normalized_axis).map_err(|_| {
                PyValueError::new_err(format!(
                    "take: axis {axis} is out of bounds for array of dimension {ndim}",
                ))
            })?;

            let mut shape =
                Vec::with_capacity(a.shape().len().saturating_sub(1) + indices_shape.len());
            shape.extend_from_slice(&a.shape()[..normalized_axis]);
            shape.extend_from_slice(&indices_shape);
            shape.extend_from_slice(&a.shape()[normalized_axis + 1..]);
            shape
        }
    };

    if result.shape() != desired_shape.as_slice() {
        let reshape_spec = desired_shape
            .iter()
            .map(|&dim| {
                isize::try_from(dim).map_err(|_| {
                    PyValueError::new_err("take: output shape exceeds signed reshape range")
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        result = result.reshape(&reshape_spec).map_err(map_ufunc_error)?;
    }

    let output = build_numpy_array_from_ufunc(py, &result)?;
    if desired_shape.is_empty() {
        return Ok(output.bind(py).get_item(PyTuple::empty(py))?.unbind());
    }

    Ok(output)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, keepdims=false))]
fn count_nonzero(
    py: Python<'_>,
    a: Py<PyAny>,
    axis: Option<Py<PyAny>>,
    keepdims: bool,
) -> PyResult<Py<PyAny>> {
    let a = extract_numeric_array(py, a.bind(py), "count_nonzero(a)")?;
    let axes = extract_axis_spec(py, axis, "count_nonzero")?;
    let axis_was_none = axes.is_none();
    let result = match axes.as_ref() {
        None => a.count_nonzero(None, keepdims),
        Some(axes) if axes.len() == 1 => a.count_nonzero(Some(axes[0]), keepdims),
        Some(axes) => a.count_nonzero_axes(axes, keepdims),
    }
    .map_err(map_ufunc_error)?;

    let output = build_numpy_array_from_ufunc(py, &result)?;
    if result.shape().is_empty() {
        if axis_was_none {
            return Ok(output.bind(py).call_method0("item")?.unbind());
        }
        return Ok(output.bind(py).get_item(())?.unbind());
    }

    Ok(output)
}

fn extract_expand_dims_axes(
    py: Python<'_>,
    axis: Py<PyAny>,
    input_ndim: usize,
) -> PyResult<Vec<isize>> {
    let Some(axes) = extract_axis_spec(py, Some(axis), "expand_dims")? else {
        return Err(PyTypeError::new_err(
            "'NoneType' object cannot be interpreted as an integer",
        ));
    };

    let new_ndim = input_ndim + axes.len();
    let mut seen = std::collections::HashSet::with_capacity(axes.len());
    for &axis in &axes {
        let normalized = if axis < 0 {
            match isize::try_from(new_ndim) {
                Ok(new_ndim) => new_ndim + axis,
                Err(_) => axis,
            }
        } else {
            axis
        };

        if normalized >= 0 && !seen.insert(normalized) {
            return Err(PyValueError::new_err("repeated axis"));
        }
    }

    Ok(axes)
}

#[pyfunction]
fn expand_dims(py: Python<'_>, a: Py<PyAny>, axis: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let a = extract_numeric_array(py, a.bind(py), "expand_dims(a)")?;
    let axes = extract_expand_dims_axes(py, axis, a.shape().len())?;
    let result = match axes.as_slice() {
        [axis] => a.expand_dims(*axis).map_err(map_ufunc_error)?,
        axes => a.expand_dims_axes(axes).map_err(map_ufunc_error)?,
    };
    build_numpy_array_from_ufunc(py, &result)
}

#[allow(dead_code)]
fn validate_trim_zeros_mode(trim: &str) -> PyResult<()> {
    if trim.is_empty() || trim.chars().any(|ch| ch != 'f' && ch != 'b') {
        return Err(PyValueError::new_err(format!(
            "unexpected character(s) in `trim`: '{trim}'"
        )));
    }

    Ok(())
}

#[pyfunction]
#[pyo3(signature = (filt, trim="fb"))]
fn trim_zeros(py: Python<'_>, filt: Py<PyAny>, trim: &str) -> PyResult<Py<PyAny>> {
    // Delegate to NumPy so list/tuple/scalar preservation and error messages
    // match trim_zeros exactly.
    let numpy = py.import("numpy")?;
    let trim_zeros_fn = numpy.getattr("trim_zeros")?;
    if trim == "fb" {
        Ok(trim_zeros_fn.call1((filt.bind(py),))?.unbind())
    } else {
        Ok(trim_zeros_fn.call1((filt.bind(py), trim))?.unbind())
    }
}

#[pyfunction]
#[pyo3(signature = (a, copy=true))]
fn masked_invalid(py: Python<'_>, a: Py<PyAny>, copy: bool) -> PyResult<Py<PyAny>> {
    let numpy = py.import("numpy")?;
    let ma = numpy.getattr("ma")?;
    let finite = numpy.getattr("isfinite")?.call1((a.bind(py),))?;
    let invalid = numpy.getattr("logical_not")?.call1((finite,))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("copy", copy)?;
    kwargs.set_item("mask", invalid)?;
    Ok(ma
        .getattr("array")?
        .call((a.bind(py),), Some(&kwargs))?
        .unbind())
}

fn minimum_fill_value_for_supported_dtype(py: Python<'_>, dtype: DType) -> PyResult<Py<PyAny>> {
    Ok(match dtype {
        DType::Bool => 1_i64.into_pyobject(py)?.into_any().unbind(),
        DType::I8 => i8::MAX.into_pyobject(py)?.into_any().unbind(),
        DType::I16 => i16::MAX.into_pyobject(py)?.into_any().unbind(),
        DType::I32 => i32::MAX.into_pyobject(py)?.into_any().unbind(),
        DType::I64 => i64::MAX.into_pyobject(py)?.into_any().unbind(),
        DType::U8 => u8::MAX.into_pyobject(py)?.into_any().unbind(),
        DType::U16 => u16::MAX.into_pyobject(py)?.into_any().unbind(),
        DType::U32 => u32::MAX.into_pyobject(py)?.into_any().unbind(),
        DType::U64 => u64::MAX.into_pyobject(py)?.into_any().unbind(),
        DType::F16 | DType::F32 | DType::F64 => {
            f64::INFINITY.into_pyobject(py)?.into_any().unbind()
        }
        DType::Complex64 | DType::Complex128 => py
            .import("builtins")?
            .getattr("complex")?
            .call1((f64::INFINITY, f64::INFINITY))?
            .unbind(),
        DType::DateTime64 | DType::TimeDelta64 => i64::MAX.into_pyobject(py)?.into_any().unbind(),
        DType::Str | DType::Structured => py.None(),
    })
}

#[pyfunction]
#[pyo3(signature = (obj,))]
fn minimum_fill_value(py: Python<'_>, obj: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let numpy = py.import("numpy")?;
    let builtins = py.import("builtins")?;
    let bound = obj.bind(py);
    let dtype = if builtins
        .call_method1("isinstance", (bound, numpy.getattr("dtype")?))?
        .extract::<bool>()?
    {
        bound.clone()
    } else {
        numpy.call_method1("asarray", (bound,))?.getattr("dtype")?
    };

    if !dtype.getattr("names")?.is_none() {
        return Ok(numpy
            .getattr("ma")?
            .getattr("minimum_fill_value")?
            .call1((bound,))?
            .unbind());
    }

    let kind = dtype.getattr("kind")?.extract::<String>()?;
    if matches!(kind.as_str(), "O" | "S" | "U" | "V") {
        return Ok(py.None());
    }

    let parsed_dtype = match kind.as_str() {
        "M" => DType::DateTime64,
        "m" => DType::TimeDelta64,
        _ => {
            let name = dtype.getattr("name")?.extract::<String>()?;
            DType::parse(&name).ok_or_else(|| {
                PyTypeError::new_err(format!("minimum_fill_value: unsupported dtype {name}"))
            })?
        }
    };

    minimum_fill_value_for_supported_dtype(py, parsed_dtype)
}

fn maximum_fill_value_for_supported_dtype(py: Python<'_>, dtype: DType) -> PyResult<Py<PyAny>> {
    // Mirror ma_maximum_fill_value_for_dtype exactly: opposite polarity of
    // minimum_fill_value. Bool → False, signed ints → MIN, unsigned ints →
    // 0, floats → -inf, complex → (-inf, -inf), datetime/timedelta →
    // -i64::MAX (one past NaT so reductions avoid the sentinel).
    Ok(match dtype {
        DType::Bool => 0_i64.into_pyobject(py)?.into_any().unbind(),
        DType::I8 => i8::MIN.into_pyobject(py)?.into_any().unbind(),
        DType::I16 => i16::MIN.into_pyobject(py)?.into_any().unbind(),
        DType::I32 => i32::MIN.into_pyobject(py)?.into_any().unbind(),
        DType::I64 => i64::MIN.into_pyobject(py)?.into_any().unbind(),
        DType::U8 => 0_u8.into_pyobject(py)?.into_any().unbind(),
        DType::U16 => 0_u16.into_pyobject(py)?.into_any().unbind(),
        DType::U32 => 0_u32.into_pyobject(py)?.into_any().unbind(),
        DType::U64 => 0_u64.into_pyobject(py)?.into_any().unbind(),
        DType::F16 | DType::F32 | DType::F64 => {
            f64::NEG_INFINITY.into_pyobject(py)?.into_any().unbind()
        }
        DType::Complex64 | DType::Complex128 => py
            .import("builtins")?
            .getattr("complex")?
            .call1((f64::NEG_INFINITY, f64::NEG_INFINITY))?
            .unbind(),
        DType::DateTime64 | DType::TimeDelta64 => {
            (-i64::MAX).into_pyobject(py)?.into_any().unbind()
        }
        DType::Str | DType::Structured => py.None(),
    })
}

#[pyfunction]
#[pyo3(signature = (obj,))]
fn maximum_fill_value(py: Python<'_>, obj: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let numpy = py.import("numpy")?;
    let builtins = py.import("builtins")?;
    let bound = obj.bind(py);
    let dtype = if builtins
        .call_method1("isinstance", (bound, numpy.getattr("dtype")?))?
        .extract::<bool>()?
    {
        bound.clone()
    } else {
        numpy.call_method1("asarray", (bound,))?.getattr("dtype")?
    };

    // Structured dtypes fall back to numpy (compound field unpacking).
    if !dtype.getattr("names")?.is_none() {
        return Ok(numpy
            .getattr("ma")?
            .getattr("maximum_fill_value")?
            .call1((bound,))?
            .unbind());
    }

    let kind = dtype.getattr("kind")?.extract::<String>()?;
    if matches!(kind.as_str(), "O" | "S" | "U" | "V") {
        return Ok(py.None());
    }

    let parsed_dtype = match kind.as_str() {
        "M" => DType::DateTime64,
        "m" => DType::TimeDelta64,
        _ => {
            let name = dtype.getattr("name")?.extract::<String>()?;
            DType::parse(&name).ok_or_else(|| {
                PyTypeError::new_err(format!("maximum_fill_value: unsupported dtype {name}"))
            })?
        }
    };

    maximum_fill_value_for_supported_dtype(py, parsed_dtype)
}

#[pyfunction]
#[pyo3(signature = (a, rcond=None, hermitian=false, **kwargs))]
fn pinv(
    py: Python<'_>,
    a: Py<PyAny>,
    rcond: Option<Py<PyAny>>,
    hermitian: bool,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let rcond = OptionalFloatKwarg::parse(py, rcond, "rcond")?;
    let rtol = parse_pinv_rtol_kwarg(py, kwargs)?;
    let array = extract_numeric_array(py, a.bind(py), "pinv(a)")?;
    let shape = array.shape();

    if shape.len() == 2
        && !matches!(array.dtype(), DType::Complex64 | DType::Complex128)
        && (!hermitian || shape[0] == shape[1])
    {
        let values = if hermitian {
            pinv_hermitian_nxn_with_tolerance_aliases(
                array.values(),
                shape[0],
                rcond.as_rcond(),
                rtol.as_rtol(),
            )
        } else {
            pinv_mxn_with_tolerance_aliases(
                array.values(),
                shape[0],
                shape[1],
                rcond.as_rcond(),
                rtol.as_rtol(),
            )
        }
        .map_err(map_ufunc_error)?;
        let result = UFuncArray::new(vec![shape[1], shape[0]], values, DType::F64)
            .map_err(map_ufunc_error)?;
        return build_numpy_array_from_ufunc(py, &result);
    }

    let numpy = py.import("numpy")?;
    let pinv_fn = numpy.getattr("linalg")?.getattr("pinv")?;
    let kwargs = PyDict::new(py);
    rcond.set_on_kwargs(&kwargs, "rcond")?;
    kwargs.set_item("hermitian", hermitian)?;
    rtol.set_on_kwargs(&kwargs, "rtol")?;
    Ok(pinv_fn.call((a.bind(py),), Some(&kwargs))?.unbind())
}

#[pyfunction]
fn eigvals(py: Python<'_>, a: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let array = extract_numeric_array(py, a.bind(py), "eigvals(a)")?;
    let shape = array.shape();
    if shape.len() == 2
        && shape[0] == shape[1]
        && !matches!(array.dtype(), DType::Complex64 | DType::Complex128)
    {
        let result = array.eigvals().map_err(map_ufunc_error)?;
        return build_numpy_eigvals_vector_from_flat_interleaved(
            py,
            result.values(),
            array.dtype(),
        );
    }

    let numpy = py.import("numpy")?;
    Ok(numpy
        .getattr("linalg")?
        .getattr("eigvals")?
        .call1((a.bind(py),))?
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (a, tol=None, hermitian=false, *, rtol=None))]
fn matrix_rank(
    py: Python<'_>,
    a: Py<PyAny>,
    tol: Option<Py<PyAny>>,
    hermitian: bool,
    rtol: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    // Passthrough to np.linalg.matrix_rank so the SVD-based rank calculation,
    // tol/rtol precedence (rtol is keyword-only, mutually-exclusive with tol),
    // hermitian fast path, and batched (..., M, N) broadcasting match numpy
    // exactly across 1-D vectors, 2-D matrices, and stacked arrays.
    let numpy = py.import("numpy")?;
    let matrix_rank_fn = numpy.getattr("linalg")?.getattr("matrix_rank")?;
    let kwargs = PyDict::new(py);
    if let Some(value) = tol {
        kwargs.set_item("tol", value.bind(py))?;
    }
    kwargs.set_item("hermitian", hermitian)?;
    if let Some(value) = rtol {
        kwargs.set_item("rtol", value.bind(py))?;
    }
    Ok(matrix_rank_fn.call((a.bind(py),), Some(&kwargs))?.unbind())
}

#[pyfunction]
#[pyo3(signature = (a, n))]
fn matrix_power(py: Python<'_>, a: Py<PyAny>, n: Py<PyAny>) -> PyResult<Py<PyAny>> {
    // Passthrough to np.linalg.matrix_power so dtype preservation for
    // nonnegative powers, inverse promotion for negative powers, and stacked
    // (..., M, M) semantics match numpy exactly.
    let numpy = py.import("numpy")?;
    Ok(numpy
        .getattr("linalg")?
        .getattr("matrix_power")?
        .call1((a.bind(py), n.bind(py)))?
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (a,))]
fn slogdet(py: Python<'_>, a: Py<PyAny>) -> PyResult<Py<PyAny>> {
    // Passthrough to np.linalg.slogdet so the (sign, logabsdet) SlogdetResult
    // namedtuple identity and batched (..., M, M) broadcasting semantics
    // match numpy exactly across real and complex inputs.
    let numpy = py.import("numpy")?;
    Ok(numpy
        .getattr("linalg")?
        .getattr("slogdet")?
        .call1((a.bind(py),))?
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (a, full_matrices=true, compute_uv=true, hermitian=false))]
fn svd(
    py: Python<'_>,
    a: Py<PyAny>,
    full_matrices: bool,
    compute_uv: bool,
    hermitian: bool,
) -> PyResult<Py<PyAny>> {
    // Passthrough to np.linalg.svd so the SVDResult namedtuple identity,
    // compute_uv=False scalar-array path, hermitian fast path, and stacked
    // (..., M, N) semantics match numpy exactly.
    let numpy = py.import("numpy")?;
    let svd_fn = numpy.getattr("linalg")?.getattr("svd")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("full_matrices", full_matrices)?;
    kwargs.set_item("compute_uv", compute_uv)?;
    kwargs.set_item("hermitian", hermitian)?;
    Ok(svd_fn.call((a.bind(py),), Some(&kwargs))?.unbind())
}

#[pyfunction]
#[pyo3(signature = (a, mode="reduced"))]
fn qr(py: Python<'_>, a: Py<PyAny>, mode: &str) -> PyResult<Py<PyAny>> {
    // Passthrough to np.linalg.qr so QRResult / ndarray / tuple return types,
    // deprecated compatibility modes, and stacked (..., M, N) semantics stay
    // byte-for-byte aligned with numpy.
    let numpy = py.import("numpy")?;
    let qr_fn = numpy.getattr("linalg")?.getattr("qr")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("mode", mode)?;
    Ok(qr_fn.call((a.bind(py),), Some(&kwargs))?.unbind())
}

#[pyfunction]
#[pyo3(signature = (*args, **kwargs))]
fn cholesky(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    match args.len() {
        0 => {
            return Err(PyTypeError::new_err(
                "cholesky() missing 1 required positional argument: 'a'",
            ));
        }
        1 => {}
        count => {
            return Err(PyTypeError::new_err(format!(
                "cholesky() takes 1 positional argument but {count} were given"
            )));
        }
    }

    // Passthrough to np.linalg.cholesky so the keyword-only upper selector,
    // lower/upper triangle semantics, complex Hermitian handling, and batched
    // (..., M, M) behavior match numpy exactly.
    let numpy = py.import("numpy")?;
    let cholesky_fn = numpy.getattr("linalg")?.getattr("cholesky")?;
    let call_kwargs = PyDict::new(py);
    let mut saw_upper = false;
    if let Some(kwargs) = kwargs {
        for (key, value) in kwargs.iter() {
            let name = key.extract::<String>()?;
            match name.as_str() {
                "upper" => {
                    call_kwargs.set_item("upper", value)?;
                    saw_upper = true;
                }
                _ => {
                    return Err(PyTypeError::new_err(format!(
                        "cholesky() got an unexpected keyword argument '{name}'"
                    )));
                }
            }
        }
    }
    if !saw_upper {
        call_kwargs.set_item("upper", false)?;
    }

    Ok(cholesky_fn
        .call((args.get_item(0)?,), Some(&call_kwargs))?
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (a, b))]
fn solve(py: Python<'_>, a: Py<PyAny>, b: Py<PyAny>) -> PyResult<Py<PyAny>> {
    // Passthrough to np.linalg.solve so square real/complex, batched
    // (..., M, M) / (..., M, K), and stacked broadcasting semantics all
    // match numpy exactly.
    let numpy = py.import("numpy")?;
    Ok(numpy
        .getattr("linalg")?
        .getattr("solve")?
        .call1((a.bind(py), b.bind(py)))?
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (a, UPLO="L"))]
#[allow(non_snake_case)]
fn eigvalsh(py: Python<'_>, a: Py<PyAny>, UPLO: &str) -> PyResult<Py<PyAny>> {
    // Passthrough to np.linalg.eigvalsh so the UPLO selector, complex-Hermitian
    // handling, and batched (..., M, M) broadcasting semantics match numpy exactly.
    let numpy = py.import("numpy")?;
    let eigvalsh_fn = numpy.getattr("linalg")?.getattr("eigvalsh")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("UPLO", UPLO)?;
    Ok(eigvalsh_fn.call((a.bind(py),), Some(&kwargs))?.unbind())
}

#[pyfunction]
#[pyo3(signature = (a,))]
fn det(py: Python<'_>, a: Py<PyAny>) -> PyResult<Py<PyAny>> {
    // Real 2-D square inputs route to fnp_linalg::det_nxn for zero-overhead
    // parity; everything else (complex, batched (..., M, M), non-2-D) is
    // passed through to np.linalg.det so numpy's broadcasting / complex
    // semantics are preserved exactly.
    let bound = a.bind(py);
    let numpy = py.import("numpy")?;
    if let Ok(array) = extract_numeric_array(py, bound, "det(a)") {
        let shape = array.shape();
        if shape.len() == 2
            && shape[0] == shape[1]
            && !matches!(array.dtype(), DType::Complex64 | DType::Complex128)
        {
            let value = fnp_linalg::det_nxn(array.values(), shape[0]).map_err(map_ufunc_error)?;
            return Ok(numpy.getattr("float64")?.call1((value,))?.unbind());
        }
    }
    Ok(numpy
        .getattr("linalg")?
        .getattr("det")?
        .call1((bound,))?
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (a,))]
fn inv(py: Python<'_>, a: Py<PyAny>) -> PyResult<Py<PyAny>> {
    // Real 2-D square inputs route to fnp_linalg::inv_nxn; complex /
    // batched / non-2-D passthrough to np.linalg.inv.
    let bound = a.bind(py);
    let numpy = py.import("numpy")?;
    if let Ok(array) = extract_numeric_array(py, bound, "inv(a)") {
        let shape = array.shape();
        if shape.len() == 2
            && shape[0] == shape[1]
            && !matches!(array.dtype(), DType::Complex64 | DType::Complex128)
        {
            let values = fnp_linalg::inv_nxn(array.values(), shape[0]).map_err(map_ufunc_error)?;
            let result = UFuncArray::new(vec![shape[0], shape[0]], values, DType::F64)
                .map_err(map_ufunc_error)?;
            return build_numpy_array_from_ufunc(py, &result);
        }
    }
    Ok(numpy
        .getattr("linalg")?
        .getattr("inv")?
        .call1((bound,))?
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (a, b, rcond=None))]
fn lstsq(
    py: Python<'_>,
    a: Py<PyAny>,
    b: Py<PyAny>,
    rcond: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    // Passthrough to np.linalg.lstsq so the 4-tuple return
    // (solution, residuals, rank, singular_values) and the rcond
    // default-handling path match numpy exactly across real/complex,
    // rank-deficient, and broadcasting inputs.
    let numpy = py.import("numpy")?;
    let lstsq_fn = numpy.getattr("linalg")?.getattr("lstsq")?;
    let kwargs = PyDict::new(py);
    if let Some(value) = rcond {
        kwargs.set_item("rcond", value.bind(py))?;
    }
    Ok(lstsq_fn
        .call((a.bind(py), b.bind(py)), Some(&kwargs))?
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (a, b, axes=None))]
fn tensorsolve(
    py: Python<'_>,
    a: Py<PyAny>,
    b: Py<PyAny>,
    axes: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    // Delegate to NumPy so axes permutation semantics and error reporting
    // stay aligned with numpy.linalg.tensorsolve.
    let numpy = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    if let Some(axes) = axes {
        kwargs.set_item("axes", axes.bind(py))?;
    }
    Ok(numpy
        .getattr("linalg")?
        .getattr("tensorsolve")?
        .call((a.bind(py), b.bind(py)), Some(&kwargs))?
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (a, ind=2))]
fn tensorinv(py: Python<'_>, a: Py<PyAny>, ind: usize) -> PyResult<Py<PyAny>> {
    let a = extract_numeric_array(py, a.bind(py), "tensorinv(a)")?;
    let result = a.tensorinv(ind).map_err(map_ufunc_error)?;
    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction]
#[pyo3(signature = (a, b, lower=false, unit_diagonal=false))]
fn solve_triangular(
    py: Python<'_>,
    a: Py<PyAny>,
    b: Py<PyAny>,
    lower: bool,
    unit_diagonal: bool,
) -> PyResult<Py<PyAny>> {
    let a = extract_numeric_array(py, a.bind(py), "solve_triangular(a)")?;
    let b = extract_numeric_array(py, b.bind(py), "solve_triangular(b)")?;
    let result = a
        .solve_triangular(&b, lower, unit_diagonal)
        .map_err(map_ufunc_error)?;
    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction]
fn isposinf(py: Python<'_>, x: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let x = extract_numeric_array(py, x.bind(py), "isposinf(x)")?;
    let result = ufunc_isposinf(&x).map_err(map_ufunc_error)?;
    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction]
fn isneginf(py: Python<'_>, x: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let x = extract_numeric_array(py, x.bind(py), "isneginf(x)")?;
    let result = ufunc_isneginf(&x).map_err(map_ufunc_error)?;
    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction]
fn signbit(py: Python<'_>, x: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let x = extract_numeric_array(py, x.bind(py), "signbit(x)")?;
    let result = ufunc_signbit(&x).map_err(map_ufunc_error)?;
    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction]
fn isnan(py: Python<'_>, x: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let x = extract_numeric_array(py, x.bind(py), "isnan(x)")?;
    build_numpy_array_from_ufunc(py, &x.elementwise_unary(UnaryOp::Isnan))
}

#[pyfunction]
fn isinf(py: Python<'_>, x: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let x = extract_numeric_array(py, x.bind(py), "isinf(x)")?;
    build_numpy_array_from_ufunc(py, &x.elementwise_unary(UnaryOp::Isinf))
}

#[pyfunction]
fn isfinite(py: Python<'_>, x: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let x = extract_numeric_array(py, x.bind(py), "isfinite(x)")?;
    build_numpy_array_from_ufunc(py, &x.elementwise_unary(UnaryOp::Isfinite))
}

#[pyfunction]
fn spacing(py: Python<'_>, x: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let x = extract_numeric_array(py, x.bind(py), "spacing(x)")?;
    let result = ufunc_spacing(&x).map_err(map_ufunc_error)?;
    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction]
fn sign(py: Python<'_>, x: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let x = extract_numeric_array(py, x.bind(py), "sign(x)")?;
    build_numpy_array_from_ufunc(py, &x.elementwise_unary(UnaryOp::Sign))
}

#[pyfunction]
fn floor(py: Python<'_>, x: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let x = extract_numeric_array(py, x.bind(py), "floor(x)")?;
    build_numpy_array_from_ufunc(py, &x.elementwise_unary(UnaryOp::Floor))
}

#[pyfunction]
fn ceil(py: Python<'_>, x: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let x = extract_numeric_array(py, x.bind(py), "ceil(x)")?;
    build_numpy_array_from_ufunc(py, &x.elementwise_unary(UnaryOp::Ceil))
}

#[pyfunction]
fn trunc(py: Python<'_>, x: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let x = extract_numeric_array(py, x.bind(py), "trunc(x)")?;
    build_numpy_array_from_ufunc(py, &x.elementwise_unary(UnaryOp::Trunc))
}

#[pyfunction]
fn rint(py: Python<'_>, x: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let x = extract_numeric_array(py, x.bind(py), "rint(x)")?;
    let result = x.elementwise_unary(UnaryOp::Rint);
    let result = match x.dtype() {
        DType::Bool => result.astype(DType::F16),
        DType::I64 | DType::U64 => result.astype(DType::F64),
        _ => result,
    };
    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction]
fn degrees(py: Python<'_>, x: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let x = extract_numeric_array(py, x.bind(py), "degrees(x)")?;
    let x = x.astype(DType::F64);
    build_numpy_array_from_ufunc(py, &x.elementwise_unary(UnaryOp::Degrees))
}

#[pyfunction]
fn radians(py: Python<'_>, x: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let x = extract_numeric_array(py, x.bind(py), "radians(x)")?;
    let x = x.astype(DType::F64);
    build_numpy_array_from_ufunc(py, &x.elementwise_unary(UnaryOp::Radians))
}

#[pyfunction]
fn sinc(py: Python<'_>, x: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let x = extract_numeric_array(py, x.bind(py), "sinc(x)")?;
    build_numpy_array_from_ufunc(py, &x.sinc())
}

#[pyfunction]
fn copysign(py: Python<'_>, x1: Py<PyAny>, x2: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let x1 = extract_numeric_array(py, x1.bind(py), "copysign(x1)")?;
    let x2 = extract_numeric_array(py, x2.bind(py), "copysign(x2)")?;
    let result = ufunc_copysign(&x1, &x2).map_err(map_ufunc_error)?;
    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction]
fn nextafter(py: Python<'_>, x1: Py<PyAny>, x2: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let x1 = extract_numeric_array(py, x1.bind(py), "nextafter(x1)")?;
    let x2 = extract_numeric_array(py, x2.bind(py), "nextafter(x2)")?;
    let result = ufunc_nextafter(&x1, &x2).map_err(map_ufunc_error)?;
    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction]
fn hypot(py: Python<'_>, x1: Py<PyAny>, x2: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let x1 = extract_numeric_array(py, x1.bind(py), "hypot(x1)")?;
    let x2 = extract_numeric_array(py, x2.bind(py), "hypot(x2)")?;
    let result = ufunc_hypot(&x1, &x2).map_err(map_ufunc_error)?;
    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction]
fn ldexp(py: Python<'_>, x1: Py<PyAny>, x2: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let x1 = extract_numeric_array(py, x1.bind(py), "ldexp(x1)")?;
    let x2 = extract_numeric_array(py, x2.bind(py), "ldexp(x2)")?;
    let result = ufunc_ldexp(&x1, &x2).map_err(map_ufunc_error)?;
    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction]
fn logaddexp(py: Python<'_>, x1: Py<PyAny>, x2: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let x1 = extract_numeric_array(py, x1.bind(py), "logaddexp(x1)")?;
    let x2 = extract_numeric_array(py, x2.bind(py), "logaddexp(x2)")?;
    let result = ufunc_logaddexp(&x1, &x2).map_err(map_ufunc_error)?;
    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction]
fn logaddexp2(py: Python<'_>, x1: Py<PyAny>, x2: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let x1 = extract_numeric_array(py, x1.bind(py), "logaddexp2(x1)")?;
    let x2 = extract_numeric_array(py, x2.bind(py), "logaddexp2(x2)")?;
    let result = ufunc_logaddexp2(&x1, &x2).map_err(map_ufunc_error)?;
    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction]
fn frexp(py: Python<'_>, x: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let x = extract_numeric_array(py, x.bind(py), "frexp(x)")?;
    let (mantissas, exponents) = ufunc_frexp(&x).map_err(map_ufunc_error)?;
    let mantissa = build_numpy_array_from_ufunc(py, &mantissas)?;
    // NumPy exposes the exponent output as an integer array.
    let exponent = build_numpy_array_from_storage(
        py,
        exponents.shape(),
        ArrayStorage::I32(
            exponents
                .values()
                .iter()
                .map(|value| *value as i32)
                .collect(),
        ),
    )?;
    Ok(PyTuple::new(py, [mantissa.bind(py), exponent.bind(py)])?
        .into_any()
        .unbind())
}

#[pyfunction]
fn modf(py: Python<'_>, x: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let x = extract_numeric_array(py, x.bind(py), "modf(x)")?;
    let (fractional, integral) = ufunc_modf(&x).map_err(map_ufunc_error)?;
    let outputs = [fractional, integral];
    build_numpy_tuple_from_ufuncs(py, &outputs)
}

#[pyfunction]
#[pyo3(signature = (x, nan=0.0, posinf=None, neginf=None))]
fn nan_to_num(
    py: Python<'_>,
    x: Py<PyAny>,
    nan: f64,
    posinf: Option<f64>,
    neginf: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let x = extract_numeric_array(py, x.bind(py), "nan_to_num(x)")?;
    let result = match (posinf, neginf) {
        (None, None) if nan == 0.0 => x.nan_to_num_default(),
        _ => x.nan_to_num(nan, posinf.unwrap_or(f64::MAX), neginf.unwrap_or(f64::MIN)),
    };
    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction]
#[pyo3(signature = (condition, a, axis=None))]
fn compress(
    py: Python<'_>,
    condition: Py<PyAny>,
    a: Py<PyAny>,
    axis: Option<isize>,
) -> PyResult<Py<PyAny>> {
    let condition = extract_condition_mask(py, condition.bind(py), "compress(condition)")?;
    let a = extract_numeric_array(py, a.bind(py), "compress(a)")?;
    let result = a.compress(&condition, axis).map_err(map_ufunc_error)?;
    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction]
fn extract(py: Python<'_>, condition: Py<PyAny>, arr: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let condition = extract_numeric_array(py, condition.bind(py), "extract(condition)")?;
    let arr = extract_numeric_array(py, arr.bind(py), "extract(arr)")?;
    let result = UFuncArray::extract(&condition, &arr).map_err(map_ufunc_error)?;
    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction]
#[pyo3(signature = (condlist, choicelist, default=None))]
fn select(
    py: Python<'_>,
    condlist: Py<PyAny>,
    choicelist: Py<PyAny>,
    default: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let condlist = extract_numeric_array_sequence(py, condlist.bind(py), "select(condlist)")?;
    let choicelist = extract_numeric_array_sequence(py, choicelist.bind(py), "select(choicelist)")?;

    if condlist.len() != choicelist.len() {
        return Err(PyValueError::new_err(
            "select: condlist and choicelist must have the same length",
        ));
    }
    if condlist.is_empty() {
        return Err(PyValueError::new_err("select: condlist must be non-empty"));
    }

    let mut result = match default {
        Some(default) => extract_numeric_array(py, default.bind(py), "select(default)")?,
        None => {
            UFuncArray::from_storage(vec![], ArrayStorage::I64(vec![0])).map_err(map_ufunc_error)?
        }
    };

    for (condition, choice) in condlist.iter().zip(choicelist.iter()).rev() {
        result = UFuncArray::where_select(condition, choice, &result).map_err(map_ufunc_error)?;
    }

    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction]
#[pyo3(signature = (a, choices, mode="raise"))]
fn choose(py: Python<'_>, a: Py<PyAny>, choices: Py<PyAny>, mode: &str) -> PyResult<Py<PyAny>> {
    let a = extract_integer_array(py, a.bind(py), "choose(a)")?;
    let choices = extract_numeric_array_sequence(py, choices.bind(py), "choose(choices)")?;
    let result = a
        .choose_with_mode(&choices, mode)
        .map_err(map_ufunc_error)?;
    build_numpy_array_from_ufunc(py, &result)
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

#[pyfunction]
#[pyo3(signature = (tup, *, dtype=None, casting="same_kind"))]
fn vstack(
    py: Python<'_>,
    tup: Py<PyAny>,
    dtype: Option<Py<PyAny>>,
    casting: &str,
) -> PyResult<Py<PyAny>> {
    if dtype
        .as_ref()
        .is_some_and(|dtype| !dtype.bind(py).is_none())
        || casting != "same_kind"
    {
        return stack_helper_numpy_fallback(
            py,
            StackHelperKind::Vertical,
            tup,
            dtype,
            Some(casting),
        );
    }
    stack_helper_default(py, tup, StackHelperKind::Vertical)
}

#[pyfunction]
#[pyo3(signature = (tup, *, dtype=None, casting="same_kind"))]
fn hstack(
    py: Python<'_>,
    tup: Py<PyAny>,
    dtype: Option<Py<PyAny>>,
    casting: &str,
) -> PyResult<Py<PyAny>> {
    if dtype
        .as_ref()
        .is_some_and(|dtype| !dtype.bind(py).is_none())
        || casting != "same_kind"
    {
        return stack_helper_numpy_fallback(
            py,
            StackHelperKind::Horizontal,
            tup,
            dtype,
            Some(casting),
        );
    }
    stack_helper_default(py, tup, StackHelperKind::Horizontal)
}

#[pyfunction]
fn dstack(py: Python<'_>, tup: Py<PyAny>) -> PyResult<Py<PyAny>> {
    stack_helper_default(py, tup, StackHelperKind::Depth)
}

#[pyfunction]
fn column_stack(py: Python<'_>, tup: Py<PyAny>) -> PyResult<Py<PyAny>> {
    stack_helper_default(py, tup, StackHelperKind::Column)
}

#[pyfunction]
#[pyo3(signature = (ary, indices_or_sections, axis=0))]
fn split(
    py: Python<'_>,
    ary: Py<PyAny>,
    indices_or_sections: Py<PyAny>,
    axis: isize,
) -> PyResult<Py<PyAny>> {
    split_helper_default(
        py,
        ary,
        indices_or_sections,
        SplitHelperKind::Equal,
        Some(axis),
    )
}

#[pyfunction]
#[pyo3(signature = (ary, indices_or_sections, axis=0))]
fn array_split(
    py: Python<'_>,
    ary: Py<PyAny>,
    indices_or_sections: Py<PyAny>,
    axis: isize,
) -> PyResult<Py<PyAny>> {
    split_helper_default(
        py,
        ary,
        indices_or_sections,
        SplitHelperKind::Flexible,
        Some(axis),
    )
}

#[pyfunction]
fn hsplit(py: Python<'_>, ary: Py<PyAny>, indices_or_sections: Py<PyAny>) -> PyResult<Py<PyAny>> {
    split_helper_default(
        py,
        ary,
        indices_or_sections,
        SplitHelperKind::Horizontal,
        None,
    )
}

#[pyfunction]
fn vsplit(py: Python<'_>, ary: Py<PyAny>, indices_or_sections: Py<PyAny>) -> PyResult<Py<PyAny>> {
    split_helper_default(
        py,
        ary,
        indices_or_sections,
        SplitHelperKind::Vertical,
        None,
    )
}

#[pyfunction]
fn dsplit(py: Python<'_>, ary: Py<PyAny>, indices_or_sections: Py<PyAny>) -> PyResult<Py<PyAny>> {
    split_helper_default(py, ary, indices_or_sections, SplitHelperKind::Depth, None)
}

#[pyfunction]
fn put(py: Python<'_>, a: Py<PyAny>, ind: Py<PyAny>, v: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let a = a.bind(py);
    require_numpy_ndarray(py, a, "put")?;

    let mut array = extract_numeric_array(py, a, "put(a)")?;
    let (_, indices) = extract_take_indices(py, ind.bind(py), "put(ind)")?;
    let values = extract_numeric_array(py, v.bind(py), "put(v)")?;

    array.put(&indices, &values).map_err(map_ufunc_error)?;
    copy_result_into_numpy_array(py, a, &array)?;
    Ok(py.None())
}

#[pyfunction]
fn place(py: Python<'_>, arr: Py<PyAny>, mask: Py<PyAny>, vals: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let arr = arr.bind(py);
    require_numpy_ndarray(py, arr, "place")?;

    let mut array = extract_numeric_array(py, arr, "place(arr)")?;
    let mask = extract_numeric_array(py, mask.bind(py), "place(mask)")?;
    let values = extract_numeric_array(py, vals.bind(py), "place(vals)")?;

    array.place(&mask, &values).map_err(map_ufunc_error)?;
    copy_result_into_numpy_array(py, arr, &array)?;
    Ok(py.None())
}

#[pyfunction]
#[pyo3(signature = (a, /, mask, values))]
fn putmask(
    py: Python<'_>,
    a: Py<PyAny>,
    mask: Py<PyAny>,
    values: Py<PyAny>,
) -> PyResult<Py<PyAny>> {
    let a = a.bind(py);
    require_numpy_ndarray(py, a, "putmask")?;

    let mut array = extract_numeric_array(py, a, "putmask(a)")?;
    let mask = extract_numeric_array(py, mask.bind(py), "putmask(mask)")?;
    let values = extract_numeric_array(py, values.bind(py), "putmask(values)")?;

    array.putmask(&mask, &values).map_err(map_ufunc_error)?;
    copy_result_into_numpy_array(py, a, &array)?;
    Ok(py.None())
}

#[pyfunction]
#[pyo3(signature = (dimensions, dtype=None))]
fn indices(
    py: Python<'_>,
    dimensions: Vec<usize>,
    dtype: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let dtype = extract_python_dtype(py, dtype, DType::I64, "indices(dtype)")?;
    let result = UFuncArray::indices(&dimensions, dtype).map_err(map_ufunc_error)?;
    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction]
#[pyo3(signature = (n, m=None, k=0, dtype=None))]
fn tri(
    py: Python<'_>,
    n: usize,
    m: Option<usize>,
    k: i64,
    dtype: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let dtype = extract_python_dtype(py, dtype, DType::F64, "tri(dtype)")?;
    let result = UFuncArray::tri(n, m, k, dtype);
    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction]
#[pyo3(signature = (condition, a, copy=true))]
fn masked_where(
    py: Python<'_>,
    condition: Py<PyAny>,
    a: Py<PyAny>,
    copy: bool,
) -> PyResult<Py<PyAny>> {
    // Passthrough to np.ma.masked_where so truthy condition broadcasting,
    // copy semantics, and the MaskedArray result type (with compatible
    // fill_value / dtype inference) match numpy exactly.
    let numpy = py.import("numpy")?;
    let masked_where_fn = numpy.getattr("ma")?.getattr("masked_where")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("copy", copy)?;
    Ok(masked_where_fn
        .call((condition.bind(py), a.bind(py)), Some(&kwargs))?
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (x, value, copy=true))]
fn masked_equal(py: Python<'_>, x: Py<PyAny>, value: Py<PyAny>, copy: bool) -> PyResult<Py<PyAny>> {
    // Passthrough to np.ma.masked_equal so equality-based masking semantics
    // (including structured dtypes and NaN handling that follows np.ma
    // conventions), copy flag forwarding, and the MaskedArray return type
    // match numpy exactly.
    let numpy = py.import("numpy")?;
    let masked_equal_fn = numpy.getattr("ma")?.getattr("masked_equal")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("copy", copy)?;
    Ok(masked_equal_fn
        .call((x.bind(py), value.bind(py)), Some(&kwargs))?
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (x, value, copy=true))]
fn masked_not_equal(
    py: Python<'_>,
    x: Py<PyAny>,
    value: Py<PyAny>,
    copy: bool,
) -> PyResult<Py<PyAny>> {
    // Passthrough to np.ma.masked_not_equal so inverse-equality masking,
    // copy flag forwarding, and the MaskedArray result type all match
    // numpy exactly across integer, float, boolean, and n-D inputs.
    let numpy = py.import("numpy")?;
    let masked_not_equal_fn = numpy.getattr("ma")?.getattr("masked_not_equal")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("copy", copy)?;
    Ok(masked_not_equal_fn
        .call((x.bind(py), value.bind(py)), Some(&kwargs))?
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (a, b))]
fn vdot(py: Python<'_>, a: Py<PyAny>, b: Py<PyAny>) -> PyResult<Py<PyAny>> {
    // Passthrough to np.vdot so the conjugate-of-first-arg semantics
    // (essential for complex inputs), input flattening, and scalar-typed
    // return all match numpy exactly across real, complex, integer, and
    // n-D inputs.
    let numpy = py.import("numpy")?;
    Ok(numpy
        .getattr("vdot")?
        .call1((a.bind(py), b.bind(py)))?
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (x, v1, v2, copy=true))]
fn masked_inside(
    py: Python<'_>,
    x: Py<PyAny>,
    v1: Py<PyAny>,
    v2: Py<PyAny>,
    copy: bool,
) -> PyResult<Py<PyAny>> {
    // Passthrough to np.ma.masked_inside so the inclusive interval
    // [min(v1,v2), max(v1,v2)] masking semantics, copy-flag forwarding,
    // and MaskedArray return type all match numpy exactly.
    let numpy = py.import("numpy")?;
    let masked_inside_fn = numpy.getattr("ma")?.getattr("masked_inside")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("copy", copy)?;
    Ok(masked_inside_fn
        .call((x.bind(py), v1.bind(py), v2.bind(py)), Some(&kwargs))?
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (x, value, copy=true))]
fn masked_greater_equal(
    py: Python<'_>,
    x: Py<PyAny>,
    value: Py<PyAny>,
    copy: bool,
) -> PyResult<Py<PyAny>> {
    // Passthrough to np.ma.masked_greater_equal: mask values >= threshold.
    // copy flag and MaskedArray return type forwarded.
    let numpy = py.import("numpy")?;
    let masked_ge_fn = numpy.getattr("ma")?.getattr("masked_greater_equal")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("copy", copy)?;
    Ok(masked_ge_fn
        .call((x.bind(py), value.bind(py)), Some(&kwargs))?
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (x, value, rtol=1e-5, atol=1e-8, copy=true, shrink=true))]
fn masked_values(
    py: Python<'_>,
    x: Py<PyAny>,
    value: Py<PyAny>,
    rtol: f64,
    atol: f64,
    copy: bool,
    shrink: bool,
) -> PyResult<Py<PyAny>> {
    // Passthrough to np.ma.masked_values: floating-point approximate-equality
    // masking using rtol/atol thresholds (mirrors np.isclose semantics for the
    // mask predicate). For integer / non-floating dtypes numpy falls back to
    // exact equality. Forwards copy and shrink kwargs.
    let numpy = py.import("numpy")?;
    let masked_values_fn = numpy.getattr("ma")?.getattr("masked_values")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("rtol", rtol)?;
    kwargs.set_item("atol", atol)?;
    kwargs.set_item("copy", copy)?;
    kwargs.set_item("shrink", shrink)?;
    Ok(masked_values_fn
        .call((x.bind(py), value.bind(py)), Some(&kwargs))?
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (x, value, copy=true))]
fn masked_less_equal(
    py: Python<'_>,
    x: Py<PyAny>,
    value: Py<PyAny>,
    copy: bool,
) -> PyResult<Py<PyAny>> {
    // Passthrough to np.ma.masked_less_equal: mask values <= threshold.
    // copy flag and MaskedArray return type forwarded.
    let numpy = py.import("numpy")?;
    let masked_le_fn = numpy.getattr("ma")?.getattr("masked_less_equal")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("copy", copy)?;
    Ok(masked_le_fn
        .call((x.bind(py), value.bind(py)), Some(&kwargs))?
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (x, v1, v2, copy=true))]
fn masked_outside(
    py: Python<'_>,
    x: Py<PyAny>,
    v1: Py<PyAny>,
    v2: Py<PyAny>,
    copy: bool,
) -> PyResult<Py<PyAny>> {
    // Passthrough to np.ma.masked_outside — the complement of
    // masked_inside. Values strictly outside [min(v1,v2), max(v1,v2)]
    // are masked; copy flag and MaskedArray return type forwarded.
    let numpy = py.import("numpy")?;
    let masked_outside_fn = numpy.getattr("ma")?.getattr("masked_outside")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("copy", copy)?;
    Ok(masked_outside_fn
        .call((x.bind(py), v1.bind(py), v2.bind(py)), Some(&kwargs))?
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (arr, axis=None))]
fn count_masked(py: Python<'_>, arr: Py<PyAny>, axis: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
    // Passthrough to np.ma.count_masked so default scalar-vs-array return
    // typing, axis reductions, and invalid-axis error text all match numpy
    // exactly for masked and unmasked inputs.
    let numpy = py.import("numpy")?;
    let count_masked_fn = numpy.getattr("ma")?.getattr("count_masked")?;
    Ok(match axis {
        Some(axis) => count_masked_fn.call1((arr.bind(py), axis.bind(py)))?,
        None => count_masked_fn.call1((arr.bind(py),))?,
    }
    .unbind())
}

#[pyfunction]
#[pyo3(signature = (a, b))]
fn kron(py: Python<'_>, a: Py<PyAny>, b: Py<PyAny>) -> PyResult<Py<PyAny>> {
    // Passthrough to np.kron so the Kronecker product output shape
    // (broadcasted from a.shape * b.shape), dtype promotion, and
    // n-dimensional broadcasting semantics match numpy exactly.
    let numpy = py.import("numpy")?;
    Ok(numpy
        .getattr("kron")?
        .call1((a.bind(py), b.bind(py)))?
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (a, b))]
fn inner(py: Python<'_>, a: Py<PyAny>, b: Py<PyAny>) -> PyResult<Py<PyAny>> {
    // Passthrough to np.inner so scalar-vs-array return typing, last-axis
    // contraction semantics, integer overflow behavior, and mismatch error
    // text all match numpy exactly.
    let numpy = py.import("numpy")?;
    Ok(numpy
        .getattr("inner")?
        .call1((a.bind(py), b.bind(py)))?
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (a, b, out=None))]
fn outer(
    py: Python<'_>,
    a: Py<PyAny>,
    b: Py<PyAny>,
    out: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    // Passthrough to np.outer so input flattening, output shape (M, N),
    // dtype promotion, and the optional `out` destination semantics match
    // numpy exactly across real/complex/integer/boolean inputs.
    let numpy = py.import("numpy")?;
    let outer_fn = numpy.getattr("outer")?;
    let kwargs = PyDict::new(py);
    if let Some(value) = out {
        kwargs.set_item("out", value.bind(py))?;
    }
    Ok(outer_fn
        .call((a.bind(py), b.bind(py)), Some(&kwargs))?
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (x, value, copy=true))]
fn masked_less(py: Python<'_>, x: Py<PyAny>, value: Py<PyAny>, copy: bool) -> PyResult<Py<PyAny>> {
    // Passthrough to np.ma.masked_less so the strict less-than masking
    // semantics, copy flag forwarding, and MaskedArray return type match
    // numpy exactly across integer / float / boolean / 2-D inputs.
    let numpy = py.import("numpy")?;
    let masked_less_fn = numpy.getattr("ma")?.getattr("masked_less")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("copy", copy)?;
    Ok(masked_less_fn
        .call((x.bind(py), value.bind(py)), Some(&kwargs))?
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (x, value, copy=true))]
fn masked_greater(
    py: Python<'_>,
    x: Py<PyAny>,
    value: Py<PyAny>,
    copy: bool,
) -> PyResult<Py<PyAny>> {
    // Passthrough to np.ma.masked_greater so the strict greater-than masking
    // semantics, copy flag forwarding, and MaskedArray return type match
    // numpy exactly across integer / float / boolean / 2-D inputs.
    let numpy = py.import("numpy")?;
    let masked_greater_fn = numpy.getattr("ma")?.getattr("masked_greater")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("copy", copy)?;
    Ok(masked_greater_fn
        .call((x.bind(py), value.bind(py)), Some(&kwargs))?
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (x, p=None))]
fn cond(py: Python<'_>, x: Py<PyAny>, p: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
    // Passthrough to np.linalg.cond so the seven valid p values
    // (None, 1, -1, 2, -2, 'fro', inf, -inf) and batched (..., M, N)
    // broadcasting match numpy exactly across real and complex inputs.
    let numpy = py.import("numpy")?;
    let cond_fn = numpy.getattr("linalg")?.getattr("cond")?;
    let kwargs = PyDict::new(py);
    if let Some(value) = p {
        kwargs.set_item("p", value.bind(py))?;
    }
    Ok(cond_fn.call((x.bind(py),), Some(&kwargs))?.unbind())
}

#[pyfunction]
#[pyo3(signature = (x, ord=None, axis=None, keepdims=false))]
fn norm(
    py: Python<'_>,
    x: Py<PyAny>,
    ord: Option<Py<PyAny>>,
    axis: Option<Py<PyAny>>,
    keepdims: bool,
) -> PyResult<Py<PyAny>> {
    // Passthrough to np.linalg.norm so ord (None/fro/nuc/int/inf/-inf/real),
    // axis (None/int/tuple), keepdims, and 1-D vector vs 2-D matrix vs
    // batched (..., M, N) broadcasting semantics all match numpy exactly.
    let numpy = py.import("numpy")?;
    let norm_fn = numpy.getattr("linalg")?.getattr("norm")?;
    let kwargs = PyDict::new(py);
    if let Some(value) = ord {
        kwargs.set_item("ord", value.bind(py))?;
    }
    if let Some(value) = axis {
        kwargs.set_item("axis", value.bind(py))?;
    }
    kwargs.set_item("keepdims", keepdims)?;
    Ok(norm_fn.call((x.bind(py),), Some(&kwargs))?.unbind())
}

#[pyfunction]
#[pyo3(signature = (x, axes=None))]
fn fftshift(py: Python<'_>, x: Py<PyAny>, axes: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
    // Passthrough to np.fft.fftshift so single-axis, multi-axis (tuple/list),
    // default all-axes, and odd/even length shift semantics match numpy
    // exactly across real and complex inputs.
    let numpy = py.import("numpy")?;
    let fftshift_fn = numpy.getattr("fft")?.getattr("fftshift")?;
    let kwargs = PyDict::new(py);
    if let Some(value) = axes {
        kwargs.set_item("axes", value.bind(py))?;
    }
    Ok(fftshift_fn.call((x.bind(py),), Some(&kwargs))?.unbind())
}

#[pyfunction]
#[pyo3(signature = (x, axes=None))]
fn ifftshift(py: Python<'_>, x: Py<PyAny>, axes: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
    // Passthrough to np.fft.ifftshift. On odd-length arrays ifftshift is the
    // true inverse of fftshift (differs from fftshift by one element).
    let numpy = py.import("numpy")?;
    let ifftshift_fn = numpy.getattr("fft")?.getattr("ifftshift")?;
    let kwargs = PyDict::new(py);
    if let Some(value) = axes {
        kwargs.set_item("axes", value.bind(py))?;
    }
    Ok(ifftshift_fn.call((x.bind(py),), Some(&kwargs))?.unbind())
}

#[pyfunction]
#[pyo3(signature = (n, d=1.0, device=None))]
fn rfftfreq(py: Python<'_>, n: usize, d: f64, device: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
    validate_cpu_device_kwarg(py, device)?;
    if n == 0 || d == 0.0 {
        return Err(PyZeroDivisionError::new_err("float division by zero"));
    }
    let result = UFuncArray::rfftfreq(n, d);
    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction]
#[pyo3(signature = (n, d=1.0, device=None))]
fn fftfreq(py: Python<'_>, n: usize, d: f64, device: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
    // Mirror the rfftfreq wrapper's device-kwarg guard and zero-division
    // checks, then delegate to the Rust UFuncArray::fftfreq fast path so
    // n/d semantics match numpy exactly for even and odd n.
    validate_cpu_device_kwarg(py, device)?;
    if n == 0 || d == 0.0 {
        return Err(PyZeroDivisionError::new_err("float division by zero"));
    }
    let result = UFuncArray::fftfreq(n, d);
    build_numpy_array_from_ufunc(py, &result)
}

fn rfft_norm_scale(norm: Option<&str>, n: usize) -> PyResult<f64> {
    match norm {
        None | Some("backward") => Ok(1.0),
        Some("ortho") => Ok(1.0 / (n as f64).sqrt()),
        Some("forward") => Ok(1.0 / n as f64),
        Some(other) => Err(PyValueError::new_err(format!(
            "Invalid norm value {other}; should be \"backward\",\"ortho\" or \"forward\"."
        ))),
    }
}

fn validate_irfft_norm(norm: Option<&str>) -> PyResult<()> {
    match norm {
        None | Some("backward") | Some("ortho") | Some("forward") => Ok(()),
        Some(other) => Err(PyValueError::new_err(format!(
            "Invalid norm value {other}; should be \"backward\",\"ortho\" or \"forward\"."
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (a, n=None, norm=None))]
fn rfft(
    py: Python<'_>,
    a: Py<PyAny>,
    n: Option<usize>,
    norm: Option<String>,
) -> PyResult<Py<PyAny>> {
    let array = extract_precise_numeric_array(py, a.bind(py), "rfft(a)")?;
    let input_n = n.unwrap_or_else(|| array.values().len());
    let result = array.rfft(n).map_err(map_ufunc_error)?;
    let scale = rfft_norm_scale(norm.as_deref(), input_n)?;
    if scale == 1.0 {
        build_numpy_complex_array_from_interleaved(py, &result)
    } else {
        let scaled_values = result.values().iter().map(|value| value * scale).collect();
        let scaled = UFuncArray::new(result.shape().to_vec(), scaled_values, result.dtype())
            .map_err(map_ufunc_error)?;
        build_numpy_complex_array_from_interleaved(py, &scaled)
    }
}

#[pyfunction]
#[pyo3(signature = (a, n=None, norm=None))]
fn irfft(
    py: Python<'_>,
    a: Py<PyAny>,
    n: Option<usize>,
    norm: Option<String>,
) -> PyResult<Py<PyAny>> {
    validate_irfft_norm(norm.as_deref())?;
    let numpy = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    if let Some(n) = n {
        kwargs.set_item("n", n)?;
    }
    if let Some(norm) = norm {
        kwargs.set_item("norm", norm)?;
    }
    Ok(numpy
        .getattr("fft")?
        .call_method("irfft", (a.bind(py),), Some(&kwargs))?
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (v, k=0))]
fn diag(py: Python<'_>, v: Py<PyAny>, k: i64) -> PyResult<Py<PyAny>> {
    let v = extract_precise_numeric_array(py, v.bind(py), "diag(v)")?;
    let result = v.diag(k).map_err(map_ufunc_error)?;
    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction]
#[pyo3(signature = (v, k=0))]
fn diagflat(py: Python<'_>, v: Py<PyAny>, k: i64) -> PyResult<Py<PyAny>> {
    let v = extract_precise_numeric_array(py, v.bind(py), "diagflat(v)")?;
    let result = v.diagflat(k);
    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction]
#[pyo3(signature = (a, offset=0, axis1=0, axis2=1))]
fn diagonal(
    py: Python<'_>,
    a: Py<PyAny>,
    offset: i64,
    axis1: isize,
    axis2: isize,
) -> PyResult<Py<PyAny>> {
    let a = extract_precise_numeric_array(py, a.bind(py), "diagonal(a)")?;
    let result = a.diagonal(offset, axis1, axis2).map_err(map_ufunc_error)?;
    build_numpy_array_from_ufunc(py, &result)
}

#[pyfunction]
#[pyo3(signature = (a, val, wrap=false))]
fn fill_diagonal(py: Python<'_>, a: Py<PyAny>, val: Py<PyAny>, wrap: bool) -> PyResult<Py<PyAny>> {
    let a = a.bind(py);
    require_numpy_ndarray(py, a, "fill_diagonal")?;

    let mut array = extract_precise_numeric_array(py, a, "fill_diagonal(a)")?;
    let values = extract_precise_numeric_array(py, val.bind(py), "fill_diagonal(val)")?;

    array
        .fill_diagonal_values(&values, wrap)
        .map_err(map_ufunc_error)?;
    copy_result_into_numpy_array(py, a, &array)?;
    Ok(py.None())
}

#[pyfunction]
#[pyo3(signature = (*args))]
fn ix_(py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    let arrays = args
        .iter()
        .enumerate()
        .map(|(index, value)| extract_ix_array(py, &value, &format!("ix_[{index}]")))
        .collect::<PyResult<Vec<_>>>()?;
    let result = UFuncArray::ix_(&arrays).map_err(map_ufunc_error)?;
    build_numpy_tuple_from_ufuncs(py, &result)
}

#[pyfunction]
#[pyo3(signature = (*xi, copy=true, sparse=false, indexing="xy"))]
fn meshgrid(
    py: Python<'_>,
    xi: &Bound<'_, PyTuple>,
    copy: bool,
    sparse: bool,
    indexing: &str,
) -> PyResult<Py<PyAny>> {
    let arrays = normalize_meshgrid_inputs(py, xi)?;
    if copy && meshgrid_rust_compatible(py, &arrays)? {
        let arrays = arrays
            .iter()
            .enumerate()
            .map(|(index, array)| {
                extract_precise_numeric_array(py, array.bind(py), &format!("meshgrid(xi[{index}])"))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let result =
            UFuncArray::meshgrid_advanced(&arrays, indexing, sparse).map_err(map_ufunc_error)?;
        return build_numpy_tuple_from_ufuncs(py, &result);
    }

    build_meshgrid_numpy_outputs(py, &arrays, indexing, sparse, copy)
}

#[pyfunction]
#[pyo3(signature = (multi_index, dims, mode=None, order="C"))]
fn ravel_multi_index(
    py: Python<'_>,
    multi_index: Py<PyAny>,
    dims: Py<PyAny>,
    mode: Option<Py<PyAny>>,
    order: &str,
) -> PyResult<Py<PyAny>> {
    let dims = extract_index_shape(py, dims.bind(py), "ravel_multi_index(dims)")?;
    let coords = extract_ravel_multi_index_inputs(py, multi_index.bind(py), dims.len())?;
    let modes = extract_ravel_multi_index_modes(py, mode)?;
    let refs: Vec<&UFuncArray> = coords.iter().collect();
    let raw_modes: Vec<&str> = modes.iter().map(String::as_str).collect();
    let result = UFuncArray::ravel_multi_index_with_options(&refs, &dims, &raw_modes, order)
        .map_err(map_ufunc_error)?;
    build_numpy_scalar_or_array_from_ufunc(py, &result)
}

#[pyfunction]
#[pyo3(signature = (indices, shape, order="C"))]
fn unravel_index(
    py: Python<'_>,
    indices: Py<PyAny>,
    shape: Py<PyAny>,
    order: &str,
) -> PyResult<Py<PyAny>> {
    let indices = extract_integer_array(py, indices.bind(py), "unravel_index(indices)")?;
    let shape = extract_index_shape(py, shape.bind(py), "unravel_index(shape)")?;
    let scalar_output = indices.shape().is_empty();
    let result =
        UFuncArray::unravel_index_order(&indices, &shape, order).map_err(map_ufunc_error)?;
    build_numpy_index_tuple_from_ufuncs(py, &result, scalar_output)
}

#[pyfunction]
#[pyo3(signature = (n, ndim=2))]
fn diag_indices(py: Python<'_>, n: usize, ndim: usize) -> PyResult<Py<PyAny>> {
    let (arrays, _) = UFuncArray::diag_indices(n, ndim);
    build_numpy_tuple_from_ufuncs(py, &arrays)
}

#[pyfunction]
fn diag_indices_from(py: Python<'_>, arr: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let shape = extract_array_shape(py, arr.bind(py), "diag_indices_from(arr)")?;
    if shape.len() < 2 {
        return Err(PyValueError::new_err("input array must be at least 2-d"));
    }

    let n = shape[0];
    if shape[1..].iter().any(|&dim| dim != n) {
        return Err(PyValueError::new_err(
            "All dimensions of input must be of equal length",
        ));
    }

    let (arrays, _) = UFuncArray::diag_indices(n, shape.len());
    build_numpy_tuple_from_ufuncs(py, &arrays)
}

#[pyfunction]
#[pyo3(signature = (n, k=0, m=None))]
fn tril_indices(py: Python<'_>, n: usize, k: i64, m: Option<usize>) -> PyResult<Py<PyAny>> {
    let (rows, cols) = UFuncArray::tril_indices(n, m.unwrap_or(n), k);
    build_numpy_tuple_from_ufuncs(py, &[rows, cols])
}

#[pyfunction]
#[pyo3(signature = (arr, k=0))]
fn tril_indices_from(py: Python<'_>, arr: Py<PyAny>, k: i64) -> PyResult<Py<PyAny>> {
    let shape = extract_array_shape(py, arr.bind(py), "tril_indices_from(arr)")?;
    if shape.len() != 2 {
        return Err(PyValueError::new_err("input array must be 2-d"));
    }

    let (rows, cols) = UFuncArray::tril_indices(shape[0], shape[1], k);
    build_numpy_tuple_from_ufuncs(py, &[rows, cols])
}

#[pyfunction]
#[pyo3(signature = (n, k=0, m=None))]
fn triu_indices(py: Python<'_>, n: usize, k: i64, m: Option<usize>) -> PyResult<Py<PyAny>> {
    let (rows, cols) = UFuncArray::triu_indices(n, m.unwrap_or(n), k);
    build_numpy_tuple_from_ufuncs(py, &[rows, cols])
}

#[pyfunction]
#[pyo3(signature = (arr, k=0))]
fn triu_indices_from(py: Python<'_>, arr: Py<PyAny>, k: i64) -> PyResult<Py<PyAny>> {
    let shape = extract_array_shape(py, arr.bind(py), "triu_indices_from(arr)")?;
    if shape.len() != 2 {
        return Err(PyValueError::new_err("input array must be 2-d"));
    }

    let (rows, cols) = UFuncArray::triu_indices(shape[0], shape[1], k);
    build_numpy_tuple_from_ufuncs(py, &[rows, cols])
}

#[pyfunction]
#[pyo3(signature = (arr, indices, values, axis=Some(-1)))]
fn put_along_axis(
    py: Python<'_>,
    arr: Py<PyAny>,
    indices: Py<PyAny>,
    values: Py<PyAny>,
    axis: Option<isize>,
) -> PyResult<Py<PyAny>> {
    let arr = arr.bind(py);
    let _ = arr.getattr("ndim")?;
    let original_shape = arr.getattr("shape")?.extract::<Vec<usize>>()?;
    let mut array = extract_numeric_array(py, arr, "put_along_axis(arr)")?;

    let indices_obj = indices.bind(py);
    let _ = indices_obj.getattr("ndim")?;
    let indices = extract_integer_array(py, indices_obj, "put_along_axis(indices)")?;
    let values = extract_numeric_array(py, values.bind(py), "put_along_axis(values)")?;

    match axis {
        Some(axis) => {
            let values = reshape_with_leading_singletons(
                values,
                array.shape().len(),
                "put_along_axis(values)",
            )?;
            array
                .put_along_axis(&indices, &values, axis)
                .map_err(map_ufunc_error)?;
        }
        None => {
            if indices.shape().len() != 1 {
                return Err(PyValueError::new_err(
                    "when axis=None, `indices` must have a single dimension.",
                ));
            }

            let values = values.flatten();
            let mut flattened = array.flatten();
            flattened
                .put_along_axis(&indices, &values, 0)
                .map_err(map_ufunc_error)?;

            let reshaped_shape = original_shape
                .iter()
                .map(|&dim| {
                    isize::try_from(dim).map_err(|_| {
                        PyValueError::new_err(format!(
                            "put_along_axis(arr): dimension {dim} exceeds signed pointer range",
                        ))
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
            array = flattened
                .reshape(&reshaped_shape)
                .map_err(map_ufunc_error)?;
        }
    }

    copy_result_into_numpy_array(py, arr, &array)?;
    Ok(py.None())
}

#[pyfunction]
#[pyo3(signature = (arr, indices, axis=Some(-1)))]
fn take_along_axis(
    py: Python<'_>,
    arr: Py<PyAny>,
    indices: Py<PyAny>,
    axis: Option<isize>,
) -> PyResult<Py<PyAny>> {
    let arr = extract_numeric_array(py, arr.bind(py), "take_along_axis(arr)")?;
    let indices = extract_integer_array(py, indices.bind(py), "take_along_axis(indices)")?;

    let result = match axis {
        Some(axis) => arr
            .take_along_axis(&indices, axis)
            .map_err(map_ufunc_error)?,
        None => {
            if indices.shape().len() != 1 {
                return Err(PyValueError::new_err(
                    "take_along_axis: when axis=None, indices must have a single dimension",
                ));
            }
            arr.flatten()
                .take_along_axis(&indices, 0)
                .map_err(map_ufunc_error)?
        }
    };

    build_numpy_array_from_ufunc(py, &result)
}

#[pymodule]
fn fnp_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_class::<PyNditerStep>()?;
    m.add_class::<PyNditer>()?;
    m.add_class::<PyFromPyFunc>()?;
    m.add_class::<PyVectorize>()?;
    m.add("mgrid", Py::new(py, PyMGridClass)?)?;
    m.add("ogrid", Py::new(py, PyOGridClass)?)?;
    m.add("r_", Py::new(py, PyRClass)?)?;
    m.add("c_", Py::new(py, PyCClass)?)?;
    m.add_function(wrap_pyfunction!(frompyfunc, m)?)?;
    m.add_function(wrap_pyfunction!(vectorize, m)?)?;
    m.add_function(wrap_pyfunction!(digitize, m)?)?;
    m.add_function(wrap_pyfunction!(bincount, m)?)?;
    m.add_function(wrap_pyfunction!(interp, m)?)?;
    m.add_function(wrap_pyfunction!(trapezoid, m)?)?;
    m.add_function(wrap_pyfunction!(trapz, m)?)?;
    m.add_function(wrap_pyfunction!(where_py, m)?)?;
    m.add_function(wrap_pyfunction!(flatnonzero, m)?)?;
    m.add_function(wrap_pyfunction!(argwhere, m)?)?;
    m.add_function(wrap_pyfunction!(count_nonzero, m)?)?;
    m.add_function(wrap_pyfunction!(expand_dims, m)?)?;
    m.add_function(wrap_pyfunction!(structured_to_unstructured, m)?)?;
    m.add_function(wrap_pyfunction!(trim_zeros, m)?)?;
    m.add_function(wrap_pyfunction!(masked_invalid, m)?)?;
    m.add_function(wrap_pyfunction!(minimum_fill_value, m)?)?;
    m.add_function(wrap_pyfunction!(maximum_fill_value, m)?)?;
    m.add_function(wrap_pyfunction!(pinv, m)?)?;
    m.add_function(wrap_pyfunction!(eigvals, m)?)?;
    m.add_function(wrap_pyfunction!(slogdet, m)?)?;
    m.add_function(wrap_pyfunction!(matrix_rank, m)?)?;
    m.add_function(wrap_pyfunction!(matrix_power, m)?)?;
    m.add_function(wrap_pyfunction!(svd, m)?)?;
    m.add_function(wrap_pyfunction!(qr, m)?)?;
    m.add_function(wrap_pyfunction!(cholesky, m)?)?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(eigvalsh, m)?)?;
    m.add_function(wrap_pyfunction!(det, m)?)?;
    m.add_function(wrap_pyfunction!(inv, m)?)?;
    m.add_function(wrap_pyfunction!(lstsq, m)?)?;
    m.add_function(wrap_pyfunction!(tensorsolve, m)?)?;
    m.add_function(wrap_pyfunction!(tensorinv, m)?)?;
    m.add_function(wrap_pyfunction!(solve_triangular, m)?)?;
    m.add_function(wrap_pyfunction!(isposinf, m)?)?;
    m.add_function(wrap_pyfunction!(isneginf, m)?)?;
    m.add_function(wrap_pyfunction!(signbit, m)?)?;
    m.add_function(wrap_pyfunction!(isnan, m)?)?;
    m.add_function(wrap_pyfunction!(isinf, m)?)?;
    m.add_function(wrap_pyfunction!(isfinite, m)?)?;
    m.add_function(wrap_pyfunction!(spacing, m)?)?;
    m.add_function(wrap_pyfunction!(sign, m)?)?;
    m.add_function(wrap_pyfunction!(floor, m)?)?;
    m.add_function(wrap_pyfunction!(ceil, m)?)?;
    m.add_function(wrap_pyfunction!(trunc, m)?)?;
    m.add_function(wrap_pyfunction!(rint, m)?)?;
    m.add_function(wrap_pyfunction!(degrees, m)?)?;
    m.add_function(wrap_pyfunction!(radians, m)?)?;
    m.add_function(wrap_pyfunction!(sinc, m)?)?;
    m.add_function(wrap_pyfunction!(copysign, m)?)?;
    m.add_function(wrap_pyfunction!(nextafter, m)?)?;
    m.add_function(wrap_pyfunction!(hypot, m)?)?;
    m.add_function(wrap_pyfunction!(ldexp, m)?)?;
    m.add_function(wrap_pyfunction!(logaddexp, m)?)?;
    m.add_function(wrap_pyfunction!(logaddexp2, m)?)?;
    m.add_function(wrap_pyfunction!(frexp, m)?)?;
    m.add_function(wrap_pyfunction!(modf, m)?)?;
    m.add_function(wrap_pyfunction!(nan_to_num, m)?)?;
    m.add_function(wrap_pyfunction!(take, m)?)?;
    m.add_function(wrap_pyfunction!(compress, m)?)?;
    m.add_function(wrap_pyfunction!(extract, m)?)?;
    m.add_function(wrap_pyfunction!(select, m)?)?;
    m.add_function(wrap_pyfunction!(choose, m)?)?;
    m.add_function(wrap_pyfunction!(searchsorted, m)?)?;
    m.add_function(wrap_pyfunction!(split, m)?)?;
    m.add_function(wrap_pyfunction!(array_split, m)?)?;
    m.add_function(wrap_pyfunction!(hsplit, m)?)?;
    m.add_function(wrap_pyfunction!(vsplit, m)?)?;
    m.add_function(wrap_pyfunction!(dsplit, m)?)?;
    m.add_function(wrap_pyfunction!(vstack, m)?)?;
    m.add_function(wrap_pyfunction!(hstack, m)?)?;
    m.add_function(wrap_pyfunction!(dstack, m)?)?;
    m.add_function(wrap_pyfunction!(column_stack, m)?)?;
    m.add_function(wrap_pyfunction!(put, m)?)?;
    m.add_function(wrap_pyfunction!(place, m)?)?;
    m.add_function(wrap_pyfunction!(putmask, m)?)?;
    m.add_function(wrap_pyfunction!(indices, m)?)?;
    m.add_function(wrap_pyfunction!(tri, m)?)?;
    m.add_function(wrap_pyfunction!(masked_where, m)?)?;
    m.add_function(wrap_pyfunction!(masked_equal, m)?)?;
    m.add_function(wrap_pyfunction!(masked_not_equal, m)?)?;
    m.add_function(wrap_pyfunction!(vdot, m)?)?;
    m.add_function(wrap_pyfunction!(masked_inside, m)?)?;
    m.add_function(wrap_pyfunction!(masked_greater_equal, m)?)?;
    m.add_function(wrap_pyfunction!(masked_values, m)?)?;
    m.add_function(wrap_pyfunction!(masked_less_equal, m)?)?;
    m.add_function(wrap_pyfunction!(masked_outside, m)?)?;
    m.add_function(wrap_pyfunction!(count_masked, m)?)?;
    m.add_function(wrap_pyfunction!(kron, m)?)?;
    m.add_function(wrap_pyfunction!(inner, m)?)?;
    m.add_function(wrap_pyfunction!(outer, m)?)?;
    m.add_function(wrap_pyfunction!(masked_less, m)?)?;
    m.add_function(wrap_pyfunction!(masked_greater, m)?)?;
    m.add_function(wrap_pyfunction!(cond, m)?)?;
    m.add_function(wrap_pyfunction!(norm, m)?)?;
    m.add_function(wrap_pyfunction!(fftshift, m)?)?;
    m.add_function(wrap_pyfunction!(ifftshift, m)?)?;
    m.add_function(wrap_pyfunction!(rfft, m)?)?;
    m.add_function(wrap_pyfunction!(irfft, m)?)?;
    m.add_function(wrap_pyfunction!(rfftfreq, m)?)?;
    m.add_function(wrap_pyfunction!(fftfreq, m)?)?;
    m.add_function(wrap_pyfunction!(diag, m)?)?;
    m.add_function(wrap_pyfunction!(diagflat, m)?)?;
    m.add_function(wrap_pyfunction!(diagonal, m)?)?;
    m.add_function(wrap_pyfunction!(fill_diagonal, m)?)?;
    m.add_function(wrap_pyfunction!(ix_, m)?)?;
    m.add_function(wrap_pyfunction!(meshgrid, m)?)?;
    m.add_function(wrap_pyfunction!(ravel_multi_index, m)?)?;
    m.add_function(wrap_pyfunction!(unravel_index, m)?)?;
    m.add_function(wrap_pyfunction!(diag_indices, m)?)?;
    m.add_function(wrap_pyfunction!(diag_indices_from, m)?)?;
    m.add_function(wrap_pyfunction!(tril_indices, m)?)?;
    m.add_function(wrap_pyfunction!(tril_indices_from, m)?)?;
    m.add_function(wrap_pyfunction!(triu_indices, m)?)?;
    m.add_function(wrap_pyfunction!(triu_indices_from, m)?)?;
    m.add_function(wrap_pyfunction!(put_along_axis, m)?)?;
    m.add_function(wrap_pyfunction!(take_along_axis, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        PyFromPyFunc, PyVectorize, argwhere, bincount, ceil, choose, compress, copysign,
        count_nonzero, degrees, diag, diag_indices, diag_indices_from, diagflat, diagonal,
        digitize, extract, fill_diagonal, flatnonzero, floor, fnp_python, frexp, hypot, indices,
        interp, isfinite, isinf, isnan, isneginf, isposinf, ix_, ldexp, logaddexp, logaddexp2,
        meshgrid, modf, nan_to_num, nextafter, place, put, put_along_axis, putmask, radians,
        ravel_multi_index, rfftfreq, rint, searchsorted, select, sign, signbit, sinc,
        solve_triangular, spacing, take, take_along_axis, tensorinv, tensorsolve, trapezoid, trapz,
        tri, tril_indices, tril_indices_from, triu_indices, triu_indices_from, trunc,
        unravel_index, where_py,
    };
    use pyo3::Bound;
    use pyo3::IntoPyObject;
    use pyo3::exceptions::{PyTypeError, PyValueError, PyZeroDivisionError};
    use pyo3::types::{
        PyAny, PyAnyMethods, PyDict, PyDictMethods, PyList, PyModule, PyTuple, PyTypeMethods,
    };
    use pyo3::{Py, PyResult, Python};

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

    fn reduce_outcome(
        py: Python<'_>,
        ufunc: &pyo3::Bound<'_, pyo3::types::PyAny>,
        array: &pyo3::Bound<'_, pyo3::types::PyAny>,
        kwargs: Option<&pyo3::Bound<'_, PyDict>>,
    ) -> PyResult<String> {
        match ufunc.call_method("reduce", (array,), kwargs) {
            Ok(value) => Ok(format!("ok:{}", repr_string(&value))),
            Err(err) => {
                let name = err.get_type(py).name()?;
                let message = err.value(py).str()?.extract::<String>()?;
                Ok(format!("err:{name}:{message}"))
            }
        }
    }

    fn call_outcome(
        py: Python<'_>,
        callable: &pyo3::Bound<'_, pyo3::types::PyAny>,
        args: &pyo3::Bound<'_, PyTuple>,
        kwargs: Option<&pyo3::Bound<'_, PyDict>>,
    ) -> PyResult<String> {
        match callable.call(args, kwargs) {
            Ok(value) => Ok(format!("ok:{}", repr_string(&value))),
            Err(err) => {
                let name = err.get_type(py).name()?;
                let message = err.value(py).str()?.extract::<String>()?;
                Ok(format!("err:{name}:{message}"))
            }
        }
    }

    fn slice_object(
        py: Python<'_>,
        start: Option<Py<PyAny>>,
        stop: Option<Py<PyAny>>,
        step: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let builtins = py.import("builtins")?;
        let start = start.unwrap_or_else(|| py.None());
        let stop = stop.unwrap_or_else(|| py.None());
        let step = step.unwrap_or_else(|| py.None());
        Ok(builtins
            .getattr("slice")?
            .call1((start.bind(py), stop.bind(py), step.bind(py)))?
            .unbind())
    }

    fn assert_array_matches_numpy(
        actual: &pyo3::Bound<'_, pyo3::types::PyAny>,
        expected: &pyo3::Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<()> {
        assert_eq!(
            actual.getattr("dtype")?.str()?.extract::<String>()?,
            expected.getattr("dtype")?.str()?.extract::<String>()?
        );
        assert_eq!(
            actual.getattr("shape")?.extract::<Vec<usize>>()?,
            expected.getattr("shape")?.extract::<Vec<usize>>()?
        );
        assert_eq!(
            repr_string(&actual.call_method0("tolist")?),
            repr_string(&expected.call_method0("tolist")?)
        );
        Ok(())
    }

    fn assert_index_tuple_matches_numpy(
        actual: &pyo3::Bound<'_, pyo3::types::PyAny>,
        expected: &pyo3::Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<()> {
        let actual_tuple = actual.downcast::<PyTuple>()?;
        let expected_tuple = expected.downcast::<PyTuple>()?;
        assert_eq!(actual_tuple.len()?, expected_tuple.len()?);

        for (actual_item, expected_item) in actual_tuple.try_iter()?.zip(expected_tuple.try_iter()?)
        {
            let actual_item = actual_item?;
            let expected_item = expected_item?;
            assert_eq!(
                actual_item.getattr("dtype")?.str()?.extract::<String>()?,
                expected_item.getattr("dtype")?.str()?.extract::<String>()?
            );
            assert_eq!(
                actual_item.getattr("shape")?.extract::<Vec<usize>>()?,
                expected_item.getattr("shape")?.extract::<Vec<usize>>()?
            );
            assert_eq!(
                repr_string(&actual_item.call_method0("tolist")?),
                repr_string(&expected_item.call_method0("tolist")?)
            );
        }

        Ok(())
    }

    fn assert_array_list_matches_numpy(
        actual: &pyo3::Bound<'_, pyo3::types::PyAny>,
        expected: &pyo3::Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<()> {
        let actual_list = actual.downcast::<PyList>()?;
        let expected_list = expected.downcast::<PyList>()?;
        assert_eq!(actual_list.len()?, expected_list.len()?);

        for (actual_item, expected_item) in actual_list.try_iter()?.zip(expected_list.try_iter()?) {
            assert_array_matches_numpy(&actual_item?, &expected_item?)?;
        }

        Ok(())
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
            assert!(module.getattr("trapezoid").is_ok());
            assert!(module.getattr("trapz").is_ok());
            assert!(module.getattr("where").is_ok());
            assert!(module.getattr("flatnonzero").is_ok());
            assert!(module.getattr("argwhere").is_ok());
            assert!(module.getattr("count_nonzero").is_ok());
            assert!(module.getattr("expand_dims").is_ok());
            assert!(module.getattr("structured_to_unstructured").is_ok());
            assert!(module.getattr("trim_zeros").is_ok());
            assert!(module.getattr("masked_invalid").is_ok());
            assert!(module.getattr("minimum_fill_value").is_ok());
            assert!(module.getattr("maximum_fill_value").is_ok());
            assert!(module.getattr("pinv").is_ok());
            assert!(module.getattr("eigvals").is_ok());
            assert!(module.getattr("masked_where").is_ok());
            assert!(module.getattr("masked_equal").is_ok());
            assert!(module.getattr("masked_not_equal").is_ok());
            assert!(module.getattr("vdot").is_ok());
            assert!(module.getattr("masked_inside").is_ok());
            assert!(module.getattr("masked_greater_equal").is_ok());
            assert!(module.getattr("masked_values").is_ok());
            assert!(module.getattr("masked_less_equal").is_ok());
            assert!(module.getattr("count_masked").is_ok());
            assert!(module.getattr("masked_outside").is_ok());
            assert!(module.getattr("kron").is_ok());
            assert!(module.getattr("inner").is_ok());
            assert!(module.getattr("outer").is_ok());
            assert!(module.getattr("masked_less").is_ok());
            assert!(module.getattr("masked_greater").is_ok());
            assert!(module.getattr("cond").is_ok());
            assert!(module.getattr("norm").is_ok());
            assert!(module.getattr("fftshift").is_ok());
            assert!(module.getattr("ifftshift").is_ok());
            assert!(module.getattr("rfft").is_ok());
            assert!(module.getattr("irfft").is_ok());
            assert!(module.getattr("slogdet").is_ok());
            assert!(module.getattr("matrix_rank").is_ok());
            assert!(module.getattr("matrix_power").is_ok());
            assert!(module.getattr("svd").is_ok());
            assert!(module.getattr("qr").is_ok());
            assert!(module.getattr("cholesky").is_ok());
            assert!(module.getattr("solve").is_ok());
            assert!(module.getattr("eigvalsh").is_ok());
            assert!(module.getattr("det").is_ok());
            assert!(module.getattr("inv").is_ok());
            assert!(module.getattr("lstsq").is_ok());
            assert!(module.getattr("tensorsolve").is_ok());
            assert!(module.getattr("tensorinv").is_ok());
            assert!(module.getattr("solve_triangular").is_ok());
            assert!(module.getattr("isposinf").is_ok());
            assert!(module.getattr("isneginf").is_ok());
            assert!(module.getattr("signbit").is_ok());
            assert!(module.getattr("isnan").is_ok());
            assert!(module.getattr("isinf").is_ok());
            assert!(module.getattr("isfinite").is_ok());
            assert!(module.getattr("spacing").is_ok());
            assert!(module.getattr("sign").is_ok());
            assert!(module.getattr("floor").is_ok());
            assert!(module.getattr("ceil").is_ok());
            assert!(module.getattr("trunc").is_ok());
            assert!(module.getattr("rint").is_ok());
            assert!(module.getattr("degrees").is_ok());
            assert!(module.getattr("radians").is_ok());
            assert!(module.getattr("sinc").is_ok());
            assert!(module.getattr("copysign").is_ok());
            assert!(module.getattr("nextafter").is_ok());
            assert!(module.getattr("hypot").is_ok());
            assert!(module.getattr("ldexp").is_ok());
            assert!(module.getattr("logaddexp").is_ok());
            assert!(module.getattr("logaddexp2").is_ok());
            assert!(module.getattr("frexp").is_ok());
            assert!(module.getattr("modf").is_ok());
            assert!(module.getattr("nan_to_num").is_ok());
            assert!(module.getattr("take").is_ok());
            assert!(module.getattr("compress").is_ok());
            assert!(module.getattr("extract").is_ok());
            assert!(module.getattr("select").is_ok());
            assert!(module.getattr("choose").is_ok());
            assert!(module.getattr("searchsorted").is_ok());
            assert!(module.getattr("split").is_ok());
            assert!(module.getattr("array_split").is_ok());
            assert!(module.getattr("hsplit").is_ok());
            assert!(module.getattr("vsplit").is_ok());
            assert!(module.getattr("dsplit").is_ok());
            assert!(module.getattr("vstack").is_ok());
            assert!(module.getattr("hstack").is_ok());
            assert!(module.getattr("dstack").is_ok());
            assert!(module.getattr("column_stack").is_ok());
            assert!(module.getattr("put").is_ok());
            assert!(module.getattr("place").is_ok());
            assert!(module.getattr("putmask").is_ok());
            assert!(module.getattr("indices").is_ok());
            assert!(module.getattr("tri").is_ok());
            assert!(module.getattr("rfftfreq").is_ok());
            assert!(module.getattr("fftfreq").is_ok());
            assert!(module.getattr("diag").is_ok());
            assert!(module.getattr("diagflat").is_ok());
            assert!(module.getattr("diagonal").is_ok());
            assert!(module.getattr("fill_diagonal").is_ok());
            assert!(module.getattr("ix_").is_ok());
            assert!(module.getattr("mgrid").is_ok());
            assert!(module.getattr("ogrid").is_ok());
            assert!(module.getattr("r_").is_ok());
            assert!(module.getattr("c_").is_ok());
            assert!(module.getattr("meshgrid").is_ok());
            assert!(module.getattr("ravel_multi_index").is_ok());
            assert!(module.getattr("unravel_index").is_ok());
            assert!(module.getattr("diag_indices").is_ok());
            assert!(module.getattr("diag_indices_from").is_ok());
            assert!(module.getattr("tril_indices").is_ok());
            assert!(module.getattr("tril_indices_from").is_ok());
            assert!(module.getattr("triu_indices").is_ok());
            assert!(module.getattr("triu_indices_from").is_ok());
            assert!(module.getattr("put_along_axis").is_ok());
            assert!(module.getattr("take_along_axis").is_ok());
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
            let ufunc = PyFromPyFunc::new_checked(callable.clone_ref(py), 1, 1, None, py)?;

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
            let ufunc = PyFromPyFunc::new_checked(callable.clone_ref(py), 2, 2, None, py)?;

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
    fn frompyfunc_live_callable_matches_numpy_zero_output_side_effects() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let callable = py.get_type::<PyList>().getattr("reverse")?.unbind();
            let actual_values = object_array(py, vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8, 9]]);
            let expected_values =
                object_array(py, vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8, 9]]);
            let ufunc = PyFromPyFunc::new_checked(callable.clone_ref(py), 1, 0, None, py)?;

            let actual_args = PyTuple::new(py, [actual_values.clone()])?;
            let actual = ufunc.call_bound(py, &actual_args)?;

            let numpy = py.import("numpy")?;
            let expected_args = PyTuple::new(py, [expected_values.clone()])?;
            let expected = numpy
                .getattr("frompyfunc")?
                .call1((callable.bind(py), 1, 0))?
                .call1(expected_args)?;

            let actual_repr = repr_string(actual.bind(py));
            let expected_repr = repr_string(&expected);
            let actual_tuple = actual.bind(py).downcast::<PyTuple>()?;
            let expected_tuple = expected.downcast::<PyTuple>()?;
            assert_eq!(actual_tuple.len()?, 0);
            assert_eq!(expected_tuple.len()?, 0);
            assert_eq!(actual_repr, expected_repr);
            assert_eq!(
                repr_string(&actual_values.call_method0("tolist")?),
                repr_string(&expected_values.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn frompyfunc_live_callable_matches_numpy_identity_and_reduce() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;

            let operator = py.import("operator")?;
            let numpy = py.import("numpy")?;
            let callable = operator.getattr("mul")?.unbind();
            let reduce_input = object_array(py, vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]]);
            let empty_input = object_array(py, Vec::<i64>::new());
            let list_input = vec![2_i64, 3_i64, 4_i64];

            let none_kwargs = PyDict::new(py);
            none_kwargs.set_item("identity", py.None())?;
            let one_kwargs = PyDict::new(py);
            one_kwargs.set_item("identity", 1_i64)?;
            let axis_kwargs = PyDict::new(py);
            axis_kwargs.set_item("axis", (0_i64, 1_i64))?;

            let actual_omitted = module
                .getattr("frompyfunc")?
                .call1((callable.bind(py), 2, 1))?;
            let expected_omitted = numpy
                .getattr("frompyfunc")?
                .call1((callable.bind(py), 2, 1))?;

            let actual_none = module
                .getattr("frompyfunc")?
                .call((callable.bind(py), 2, 1), Some(&none_kwargs))?;
            let expected_none = numpy
                .getattr("frompyfunc")?
                .call((callable.bind(py), 2, 1), Some(&none_kwargs))?;

            let actual_one = module
                .getattr("frompyfunc")?
                .call((callable.bind(py), 2, 1), Some(&one_kwargs))?;
            let expected_one = numpy
                .getattr("frompyfunc")?
                .call((callable.bind(py), 2, 1), Some(&one_kwargs))?;

            for (actual, expected) in [
                (&actual_omitted, &expected_omitted),
                (&actual_none, &expected_none),
                (&actual_one, &expected_one),
            ] {
                assert_eq!(
                    repr_string(&actual.getattr("identity")?),
                    repr_string(&expected.getattr("identity")?)
                );
            }

            for (actual, expected) in [
                (&actual_omitted, &expected_omitted),
                (&actual_none, &expected_none),
                (&actual_one, &expected_one),
            ] {
                assert_eq!(
                    reduce_outcome(py, actual, &object_array(py, list_input.clone()), None)?,
                    reduce_outcome(py, expected, &object_array(py, list_input.clone()), None)?,
                );
            }

            assert_eq!(
                reduce_outcome(py, &actual_omitted, &reduce_input, Some(&axis_kwargs))?,
                reduce_outcome(py, &expected_omitted, &reduce_input, Some(&axis_kwargs))?,
            );
            assert_eq!(
                reduce_outcome(py, &actual_none, &reduce_input, Some(&axis_kwargs))?,
                reduce_outcome(py, &expected_none, &reduce_input, Some(&axis_kwargs))?,
            );
            assert_eq!(
                reduce_outcome(py, &actual_one, &reduce_input, Some(&axis_kwargs))?,
                reduce_outcome(py, &expected_one, &reduce_input, Some(&axis_kwargs))?,
            );

            assert_eq!(
                reduce_outcome(py, &actual_omitted, &empty_input, None)?,
                reduce_outcome(py, &expected_omitted, &empty_input, None)?,
            );
            assert_eq!(
                reduce_outcome(py, &actual_none, &empty_input, None)?,
                reduce_outcome(py, &expected_none, &empty_input, None)?,
            );
            assert_eq!(
                reduce_outcome(py, &actual_one, &empty_input, None)?,
                reduce_outcome(py, &expected_one, &empty_input, None)?,
            );

            Ok(())
        });
    }

    #[test]
    fn frompyfunc_live_callable_matches_numpy_reduce_kwargs_and_out() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;

            let operator = py.import("operator")?;
            let numpy = py.import("numpy")?;
            let callable = operator.getattr("mul")?.unbind();
            let reduce_input = object_array(
                py,
                vec![vec![2_i64, 3_i64, 4_i64], vec![5_i64, 6_i64, 7_i64]],
            );

            let actual = module
                .getattr("frompyfunc")?
                .call1((callable.bind(py), 2, 1))?;
            let expected = numpy
                .getattr("frompyfunc")?
                .call1((callable.bind(py), 2, 1))?;

            let keepdims_kwargs = PyDict::new(py);
            keepdims_kwargs.set_item("axis", 1_i64)?;
            keepdims_kwargs.set_item("keepdims", true)?;
            assert_eq!(
                reduce_outcome(py, &actual, &reduce_input, Some(&keepdims_kwargs))?,
                reduce_outcome(py, &expected, &reduce_input, Some(&keepdims_kwargs))?,
            );

            let where_mask = numeric_array(py, vec![true, false, true], "bool");
            let where_kwargs = PyDict::new(py);
            where_kwargs.set_item("axis", 0_i64)?;
            where_kwargs.set_item("where", where_mask.clone())?;
            where_kwargs.set_item("initial", 11_i64)?;
            assert_eq!(
                reduce_outcome(py, &actual, &reduce_input, Some(&where_kwargs))?,
                reduce_outcome(py, &expected, &reduce_input, Some(&where_kwargs))?,
            );

            let where_error_kwargs = PyDict::new(py);
            where_error_kwargs.set_item("axis", 0_i64)?;
            where_error_kwargs.set_item("where", where_mask)?;
            assert_eq!(
                reduce_outcome(py, &actual, &reduce_input, Some(&where_error_kwargs))?,
                reduce_outcome(py, &expected, &reduce_input, Some(&where_error_kwargs))?,
            );

            let actual_out = object_array(py, vec![0_i64, 0_i64, 0_i64]);
            let expected_out = object_array(py, vec![0_i64, 0_i64, 0_i64]);
            let actual_out_kwargs = PyDict::new(py);
            let expected_out_kwargs = PyDict::new(py);
            actual_out_kwargs.set_item("axis", 0_i64)?;
            expected_out_kwargs.set_item("axis", 0_i64)?;
            actual_out_kwargs.set_item("out", PyTuple::new(py, [actual_out.clone()])?)?;
            expected_out_kwargs.set_item("out", PyTuple::new(py, [expected_out.clone()])?)?;

            let actual_result =
                actual.call_method("reduce", (reduce_input.clone(),), Some(&actual_out_kwargs))?;
            let expected_result =
                expected.call_method("reduce", (reduce_input,), Some(&expected_out_kwargs))?;

            assert_array_matches_numpy(&actual_result, &expected_result)?;
            assert_array_matches_numpy(&actual_out, &expected_out)?;
            Ok(())
        });
    }

    #[test]
    fn frompyfunc_live_callable_matches_numpy_reduce_scalar_where_and_direct_out() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;

            let operator = py.import("operator")?;
            let numpy = py.import("numpy")?;
            let callable = operator.getattr("mul")?.unbind();
            let reduce_input = object_array(
                py,
                vec![vec![2_i64, 3_i64, 4_i64], vec![5_i64, 6_i64, 7_i64]],
            );

            let actual = module
                .getattr("frompyfunc")?
                .call1((callable.bind(py), 2, 1))?;
            let expected = numpy
                .getattr("frompyfunc")?
                .call1((callable.bind(py), 2, 1))?;

            let where_true_kwargs = PyDict::new(py);
            where_true_kwargs.set_item("axis", 0_i64)?;
            where_true_kwargs.set_item("where", true)?;
            assert_eq!(
                reduce_outcome(py, &actual, &reduce_input, Some(&where_true_kwargs))?,
                reduce_outcome(py, &expected, &reduce_input, Some(&where_true_kwargs))?,
            );

            let where_false_initial_kwargs = PyDict::new(py);
            where_false_initial_kwargs.set_item("axis", 0_i64)?;
            where_false_initial_kwargs.set_item("where", false)?;
            where_false_initial_kwargs.set_item("initial", 11_i64)?;
            assert_eq!(
                reduce_outcome(
                    py,
                    &actual,
                    &reduce_input,
                    Some(&where_false_initial_kwargs)
                )?,
                reduce_outcome(
                    py,
                    &expected,
                    &reduce_input,
                    Some(&where_false_initial_kwargs)
                )?,
            );

            let where_false_error_kwargs = PyDict::new(py);
            where_false_error_kwargs.set_item("axis", 0_i64)?;
            where_false_error_kwargs.set_item("where", false)?;
            assert_eq!(
                reduce_outcome(py, &actual, &reduce_input, Some(&where_false_error_kwargs))?,
                reduce_outcome(
                    py,
                    &expected,
                    &reduce_input,
                    Some(&where_false_error_kwargs)
                )?,
            );

            let actual_out = object_array(py, vec![0_i64, 0_i64, 0_i64]);
            let expected_out = object_array(py, vec![0_i64, 0_i64, 0_i64]);
            let actual_out_kwargs = PyDict::new(py);
            let expected_out_kwargs = PyDict::new(py);
            actual_out_kwargs.set_item("axis", 0_i64)?;
            expected_out_kwargs.set_item("axis", 0_i64)?;
            actual_out_kwargs.set_item("out", actual_out.clone())?;
            expected_out_kwargs.set_item("out", expected_out.clone())?;

            let actual_result =
                actual.call_method("reduce", (reduce_input.clone(),), Some(&actual_out_kwargs))?;
            let expected_result = expected.call_method(
                "reduce",
                (reduce_input.clone(),),
                Some(&expected_out_kwargs),
            )?;

            assert_array_matches_numpy(&actual_result, &expected_result)?;
            assert_array_matches_numpy(&actual_out, &expected_out)?;

            let actual_bad_out = object_array(py, vec![0_i64, 0_i64, 0_i64]);
            let expected_bad_out = object_array(py, vec![0_i64, 0_i64, 0_i64]);
            let actual_bad_out_kwargs = PyDict::new(py);
            let expected_bad_out_kwargs = PyDict::new(py);
            actual_bad_out_kwargs.set_item("axis", 0_i64)?;
            expected_bad_out_kwargs.set_item("axis", 0_i64)?;
            actual_bad_out_kwargs.set_item("out", PyList::new(py, [actual_bad_out])?)?;
            expected_bad_out_kwargs.set_item("out", PyList::new(py, [expected_bad_out])?)?;

            assert_eq!(
                reduce_outcome(py, &actual, &reduce_input, Some(&actual_bad_out_kwargs))?,
                reduce_outcome(py, &expected, &reduce_input, Some(&expected_bad_out_kwargs))?,
            );

            Ok(())
        });
    }

    #[test]
    fn frompyfunc_live_callable_matches_numpy_reduce_negative_axis() {
        // Locks parity for frompyfunc.reduce() when the axis is specified as a
        // negative integer (-1/-2) or a single-element tuple containing a
        // negative integer — both forms are accepted by NumPy's underlying
        // reduce path and collapse to the same axis as their positive
        // counterpart. Multi-axis negative tuples are not exercised here
        // because NumPy rejects them for non-reorderable ufuncs (covered via
        // the existing error-parity test in identity_and_reduce).
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;

            let operator = py.import("operator")?;
            let numpy = py.import("numpy")?;
            let callable = operator.getattr("add")?.unbind();
            let reduce_input = object_array(
                py,
                vec![vec![1_i64, 2_i64, 3_i64], vec![4_i64, 5_i64, 6_i64]],
            );

            let actual = module
                .getattr("frompyfunc")?
                .call1((callable.bind(py), 2, 1))?;
            let expected = numpy
                .getattr("frompyfunc")?
                .call1((callable.bind(py), 2, 1))?;

            for axis_value in [-1_i64, -2_i64] {
                let kwargs = PyDict::new(py);
                kwargs.set_item("axis", axis_value)?;
                assert_eq!(
                    reduce_outcome(py, &actual, &reduce_input, Some(&kwargs))?,
                    reduce_outcome(py, &expected, &reduce_input, Some(&kwargs))?,
                    "axis={axis_value}",
                );
            }

            for axis_value in [-1_i64, -2_i64] {
                let kwargs = PyDict::new(py);
                let axis_tuple = PyTuple::new(py, [axis_value])?;
                kwargs.set_item("axis", axis_tuple)?;
                assert_eq!(
                    reduce_outcome(py, &actual, &reduce_input, Some(&kwargs))?,
                    reduce_outcome(py, &expected, &reduce_input, Some(&kwargs))?,
                    "axis=({axis_value},)",
                );
            }

            let kwargs = PyDict::new(py);
            kwargs.set_item("axis", -1_i64)?;
            kwargs.set_item("keepdims", true)?;
            assert_eq!(
                reduce_outcome(py, &actual, &reduce_input, Some(&kwargs))?,
                reduce_outcome(py, &expected, &reduce_input, Some(&kwargs))?,
                "axis=-1 keepdims=True",
            );

            Ok(())
        });
    }

    #[test]
    fn frompyfunc_reduce_does_not_delegate_to_numpy_frompyfunc() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;

            let numpy = py.import("numpy")?;
            let operator = py.import("operator")?;
            let original = numpy.getattr("frompyfunc")?.unbind();
            let poison = PyModule::from_code(
                py,
                pyo3::ffi::c_str!(
                    "def fail(*args, **kwargs):\n    raise RuntimeError('numpy.frompyfunc should not be called')\n"
                ),
                pyo3::ffi::c_str!("poison_frompyfunc.py"),
                pyo3::ffi::c_str!("poison_frompyfunc"),
            )?;

            let result = (|| -> PyResult<()> {
                numpy.setattr("frompyfunc", poison.getattr("fail")?)?;
                let ufunc =
                    module
                        .getattr("frompyfunc")?
                        .call1((operator.getattr("mul")?, 2, 1))?;

                assert_eq!(repr_string(&ufunc.getattr("identity")?), "None");
                assert_eq!(
                    reduce_outcome(
                        py,
                        &ufunc,
                        &object_array(py, vec![2_i64, 3_i64, 4_i64]),
                        None
                    )?,
                    "ok:24"
                );
                Ok(())
            })();

            numpy.setattr("frompyfunc", original.bind(py))?;
            result
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
    fn bincount_matches_numpy_defaults_weights_and_minlength() {
        // Live-NumPy conformance for the newly exposed fnp-python bincount
        // binding. Covers default (count) mode, weighted sums (F64 output),
        // minlength padding beyond max(x)+1, minlength smaller than max+1
        // (max wins), empty input with minlength, and the length-mismatch
        // error path.
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }
            let numpy = py.import("numpy")?;

            // Default: counts, int64 output
            let x = numeric_array(py, vec![0, 1, 1, 2, 2, 2, 3], "int64");
            let actual = bincount(py, x.clone().unbind(), None, 0)?;
            let expected = numpy.call_method1("bincount", (x.clone(),))?;
            assert_eq!(
                actual
                    .bind(py)
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?,
                expected.getattr("dtype")?.str()?.extract::<String>()?
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );

            // Weighted sums, float64 output
            let x_w = numeric_array(py, vec![0, 1, 1, 2], "int64");
            let weights = numeric_array(py, vec![1.0, 2.0, 3.0, 4.0], "float64");
            let actual_w = bincount(py, x_w.clone().unbind(), Some(weights.clone().unbind()), 0)?;
            let kwargs_w = PyDict::new(py);
            kwargs_w.set_item("weights", weights)?;
            let expected_w = numpy.call_method("bincount", (x_w.clone(),), Some(&kwargs_w))?;
            assert_eq!(
                actual_w
                    .bind(py)
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?,
                expected_w.getattr("dtype")?.str()?.extract::<String>()?
            );
            assert_eq!(
                repr_string(&actual_w.bind(py).call_method0("tolist")?),
                repr_string(&expected_w.call_method0("tolist")?)
            );

            // minlength larger than max(x)+1 pads with zeros
            let x_pad = numeric_array(py, vec![0, 1, 2], "int64");
            let actual_pad = bincount(py, x_pad.clone().unbind(), None, 5)?;
            let kwargs_pad = PyDict::new(py);
            kwargs_pad.set_item("minlength", 5)?;
            let expected_pad =
                numpy.call_method("bincount", (x_pad.clone(),), Some(&kwargs_pad))?;
            assert_eq!(
                repr_string(&actual_pad.bind(py).call_method0("tolist")?),
                repr_string(&expected_pad.call_method0("tolist")?)
            );

            // minlength smaller than max(x)+1 is ignored; max wins
            let x_max = numeric_array(py, vec![0, 5], "int64");
            let actual_max = bincount(py, x_max.clone().unbind(), None, 2)?;
            let kwargs_max = PyDict::new(py);
            kwargs_max.set_item("minlength", 2)?;
            let expected_max =
                numpy.call_method("bincount", (x_max.clone(),), Some(&kwargs_max))?;
            assert_eq!(
                repr_string(&actual_max.bind(py).call_method0("tolist")?),
                repr_string(&expected_max.call_method0("tolist")?)
            );

            // Empty input with minlength returns zeros of that length
            let empty = numeric_array(py, Vec::<i64>::new(), "int64");
            let actual_empty = bincount(py, empty.clone().unbind(), None, 3)?;
            let kwargs_empty = PyDict::new(py);
            kwargs_empty.set_item("minlength", 3)?;
            let expected_empty = numpy.call_method("bincount", (empty,), Some(&kwargs_empty))?;
            assert_eq!(
                repr_string(&actual_empty.bind(py).call_method0("tolist")?),
                repr_string(&expected_empty.call_method0("tolist")?)
            );

            // Length-mismatch weights raise ValueError in NumPy; we propagate
            // the same error category via UFuncError::Msg.
            let x_mm = numeric_array(py, vec![0, 1], "int64");
            let w_mm = numeric_array(py, vec![1.0, 2.0, 3.0], "float64");
            let err = bincount(py, x_mm.unbind(), Some(w_mm.unbind()), 0).unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains("weights") && msg.contains("length"),
                "unexpected error: {msg}"
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
    fn trapezoid_matches_numpy_scalar_result() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let y = numeric_array(py, vec![1.0, 2.0, 4.0, 8.0], "float64");
            let actual = trapezoid(py, y.clone().unbind(), None, 1.0, -1)?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("trapezoid", (y,))?;

            assert_eq!(repr_string(actual.bind(py)), repr_string(&expected));
            Ok(())
        });
    }

    #[test]
    fn trapezoid_matches_numpy_dx_and_axis() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let y = numeric_array(
                py,
                vec![vec![1.0, 2.0, 4.0], vec![3.0, 5.0, 9.0]],
                "float64",
            );
            let actual = trapezoid(py, y.clone().unbind(), None, 0.5, 0)?;
            let numpy = py.import("numpy")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("dx", 0.5)?;
            kwargs.set_item("axis", 0)?;
            let expected = numpy.call_method("trapezoid", (y,), Some(&kwargs))?;

            assert_array_matches_numpy(actual.bind(py), &expected)?;
            Ok(())
        });
    }

    #[test]
    fn trapezoid_matches_numpy_with_x_spacing() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let y = numeric_array(
                py,
                vec![vec![1.0, 2.0, 4.0], vec![3.0, 5.0, 9.0]],
                "float64",
            );
            let x = numeric_array(py, vec![0.0, 1.0, 3.0], "float64");
            let actual = trapezoid(py, y.clone().unbind(), Some(x.clone().unbind()), 99.0, 1)?;
            let numpy = py.import("numpy")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("x", x)?;
            kwargs.set_item("dx", 99.0)?;
            kwargs.set_item("axis", 1)?;
            let expected = numpy.call_method("trapezoid", (y,), Some(&kwargs))?;

            assert_array_matches_numpy(actual.bind(py), &expected)?;
            Ok(())
        });
    }

    #[test]
    fn trapezoid_matches_numpy_with_broadcast_x_spacing() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let y = numeric_array(
                py,
                vec![vec![1.0, 2.0, 4.0], vec![3.0, 5.0, 9.0]],
                "float64",
            );
            let x = numeric_array(py, vec![vec![0.0, 1.0, 3.0]], "float64");
            let actual = trapezoid(py, y.clone().unbind(), Some(x.clone().unbind()), 1.0, 1)?;
            let numpy = py.import("numpy")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("x", x)?;
            kwargs.set_item("axis", 1)?;
            let expected = numpy.call_method("trapezoid", (y,), Some(&kwargs))?;

            assert_array_matches_numpy(actual.bind(py), &expected)?;
            Ok(())
        });
    }

    #[test]
    fn trapezoid_matches_numpy_with_same_shape_x_spacing() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let y = numeric_array(
                py,
                vec![vec![1.0, 2.0, 4.0], vec![3.0, 5.0, 9.0]],
                "float64",
            );
            let x = numeric_array(
                py,
                vec![vec![0.0, 1.0, 3.0], vec![0.0, 2.0, 5.0]],
                "float64",
            );
            let actual = trapezoid(py, y.clone().unbind(), Some(x.clone().unbind()), 1.0, 1)?;
            let numpy = py.import("numpy")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("x", x)?;
            kwargs.set_item("axis", 1)?;
            let expected = numpy.call_method("trapezoid", (y,), Some(&kwargs))?;

            assert_array_matches_numpy(actual.bind(py), &expected)?;
            Ok(())
        });
    }

    #[test]
    fn trapz_alias_matches_numpy_trapezoid() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let y = numeric_array(py, vec![2.0, 3.0, 7.0], "float64");
            let actual = trapz(py, y.clone().unbind(), None, 2.0, -1)?;
            let numpy = py.import("numpy")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("dx", 2.0)?;
            let expected = numpy.call_method("trapezoid", (y,), Some(&kwargs))?;

            assert_eq!(repr_string(actual.bind(py)), repr_string(&expected));
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
    fn flatnonzero_matches_numpy_multidimensional_input() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let arr = numeric_array(
                py,
                vec![vec![0_i64, 2_i64, 0_i64], vec![3_i64, 0_i64, 4_i64]],
                "int64",
            );
            let actual = flatnonzero(py, arr.clone().unbind())?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("flatnonzero", (arr,))?;

            assert_eq!(
                actual
                    .bind(py)
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?,
                expected.getattr("dtype")?.str()?.extract::<String>()?
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn flatnonzero_matches_numpy_scalar_input() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let arr = numeric_array(py, 1_i64, "int64");
            let actual = flatnonzero(py, arr.clone().unbind())?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("flatnonzero", (arr,))?;

            assert_eq!(
                actual.bind(py).getattr("shape")?.extract::<Vec<usize>>()?,
                expected.getattr("shape")?.extract::<Vec<usize>>()?
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn flatnonzero_matches_numpy_zero_scalar_and_empty_input() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let numpy = py.import("numpy")?;

            let zero_scalar = numeric_array(py, 0_i64, "int64");
            let zero_actual = flatnonzero(py, zero_scalar.clone().unbind())?;
            let zero_expected = numpy.call_method1("flatnonzero", (zero_scalar,))?;

            assert_eq!(
                zero_actual
                    .bind(py)
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?,
                zero_expected.getattr("dtype")?.str()?.extract::<String>()?
            );
            assert_eq!(
                zero_actual
                    .bind(py)
                    .getattr("shape")?
                    .extract::<Vec<usize>>()?,
                zero_expected.getattr("shape")?.extract::<Vec<usize>>()?
            );
            assert_eq!(
                repr_string(&zero_actual.bind(py).call_method0("tolist")?),
                repr_string(&zero_expected.call_method0("tolist")?)
            );

            let empty = numeric_array(py, Vec::<i64>::new(), "int64");
            let empty_actual = flatnonzero(py, empty.clone().unbind())?;
            let empty_expected = numpy.call_method1("flatnonzero", (empty,))?;

            assert_eq!(
                empty_actual
                    .bind(py)
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?,
                empty_expected
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?
            );
            assert_eq!(
                empty_actual
                    .bind(py)
                    .getattr("shape")?
                    .extract::<Vec<usize>>()?,
                empty_expected.getattr("shape")?.extract::<Vec<usize>>()?
            );
            assert_eq!(
                repr_string(&empty_actual.bind(py).call_method0("tolist")?),
                repr_string(&empty_expected.call_method0("tolist")?)
            );

            Ok(())
        });
    }

    #[test]
    fn argwhere_matches_numpy_multidimensional_input() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let arr = numeric_array(
                py,
                vec![vec![0.0, 2.5, 0.0], vec![3.5, 0.0, -4.0]],
                "float64",
            );
            let actual = argwhere(py, arr.clone().unbind())?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("argwhere", (arr,))?;

            assert_eq!(
                actual.bind(py).getattr("shape")?.extract::<Vec<usize>>()?,
                expected.getattr("shape")?.extract::<Vec<usize>>()?
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn argwhere_matches_numpy_scalar_nonzero_shape() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let arr = numeric_array(py, 1_i64, "int64");
            let actual = argwhere(py, arr.clone().unbind())?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("argwhere", (arr,))?;

            assert_eq!(
                actual.bind(py).getattr("shape")?.extract::<Vec<usize>>()?,
                expected.getattr("shape")?.extract::<Vec<usize>>()?
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn argwhere_matches_numpy_zero_scalar_and_empty_input() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let numpy = py.import("numpy")?;

            let zero_scalar = numeric_array(py, 0_i64, "int64");
            let zero_actual = argwhere(py, zero_scalar.clone().unbind())?;
            let zero_expected = numpy.call_method1("argwhere", (zero_scalar,))?;

            assert_eq!(
                zero_actual
                    .bind(py)
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?,
                zero_expected.getattr("dtype")?.str()?.extract::<String>()?
            );
            assert_eq!(
                zero_actual
                    .bind(py)
                    .getattr("shape")?
                    .extract::<Vec<usize>>()?,
                zero_expected.getattr("shape")?.extract::<Vec<usize>>()?
            );
            assert_eq!(
                repr_string(&zero_actual.bind(py).call_method0("tolist")?),
                repr_string(&zero_expected.call_method0("tolist")?)
            );

            let empty = numeric_array(py, Vec::<i64>::new(), "int64");
            let empty_actual = argwhere(py, empty.clone().unbind())?;
            let empty_expected = numpy.call_method1("argwhere", (empty,))?;

            assert_eq!(
                empty_actual
                    .bind(py)
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?,
                empty_expected
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?
            );
            assert_eq!(
                empty_actual
                    .bind(py)
                    .getattr("shape")?
                    .extract::<Vec<usize>>()?,
                empty_expected.getattr("shape")?.extract::<Vec<usize>>()?
            );
            assert_eq!(
                repr_string(&empty_actual.bind(py).call_method0("tolist")?),
                repr_string(&empty_expected.call_method0("tolist")?)
            );

            Ok(())
        });
    }

    #[test]
    fn count_nonzero_matches_numpy_axis_none() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let arr = numeric_array(
                py,
                vec![vec![1_i64, 0_i64, 3_i64], vec![0_i64, 5_i64, 0_i64]],
                "int64",
            );
            let actual = count_nonzero(py, arr.clone().unbind(), None, false)?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("count_nonzero", (arr,))?;

            assert_eq!(repr_string(actual.bind(py)), repr_string(&expected));
            Ok(())
        });
    }

    #[test]
    fn count_nonzero_matches_numpy_axis_and_keepdims() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let arr = numeric_array(
                py,
                vec![vec![1_i64, 0_i64, 3_i64], vec![0_i64, 5_i64, 0_i64]],
                "int64",
            );
            let axis = 1_i32.into_pyobject(py)?.unbind();
            let actual = count_nonzero(py, arr.clone().unbind(), Some(axis.into()), true)?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method(
                "count_nonzero",
                (arr,),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("axis", 1)?;
                    kwargs.set_item("keepdims", true)?;
                    kwargs
                }),
            )?;

            assert_eq!(
                actual.bind(py).getattr("shape")?.extract::<Vec<usize>>()?,
                expected.getattr("shape")?.extract::<Vec<usize>>()?
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn count_nonzero_matches_numpy_axis_tuple_and_empty_tuple() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let arr = numeric_array(
                py,
                vec![vec![1_i64, 0_i64, 3_i64], vec![0_i64, 5_i64, 0_i64]],
                "int64",
            );
            let axis_tuple = PyTuple::new(py, [0_isize, 1_isize])?.unbind();
            let actual_tuple = count_nonzero(
                py,
                arr.clone().unbind(),
                Some(axis_tuple.clone_ref(py).into()),
                false,
            )?;
            let numpy = py.import("numpy")?;
            let expected_tuple =
                numpy.call_method1("count_nonzero", (arr.clone(), axis_tuple.bind(py)))?;

            assert_eq!(
                repr_string(actual_tuple.bind(py)),
                repr_string(&expected_tuple)
            );

            let empty_axis = PyTuple::empty(py).unbind();
            let actual_empty = count_nonzero(
                py,
                arr.clone().unbind(),
                Some(empty_axis.clone_ref(py).into()),
                false,
            )?;
            let expected_empty = numpy.call_method1("count_nonzero", (arr, empty_axis.bind(py)))?;

            assert_eq!(
                actual_empty
                    .bind(py)
                    .getattr("shape")?
                    .extract::<Vec<usize>>()?,
                expected_empty.getattr("shape")?.extract::<Vec<usize>>()?
            );
            assert_eq!(
                repr_string(&actual_empty.bind(py).call_method0("tolist")?),
                repr_string(&expected_empty.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn count_nonzero_matches_numpy_keepdims_none_and_scalar_inputs() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let numpy = py.import("numpy")?;
            let array = numeric_array(
                py,
                vec![vec![1_i64, 0_i64, 3_i64], vec![0_i64, 5_i64, 0_i64]],
                "int64",
            );
            let actual_keepdims = count_nonzero(py, array.clone().unbind(), None, true)?;
            let expected_keepdims = numpy.call_method(
                "count_nonzero",
                (array,),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("keepdims", true)?;
                    kwargs
                }),
            )?;

            assert_eq!(
                actual_keepdims
                    .bind(py)
                    .getattr("shape")?
                    .extract::<Vec<usize>>()?,
                expected_keepdims
                    .getattr("shape")?
                    .extract::<Vec<usize>>()?
            );
            assert_eq!(
                repr_string(&actual_keepdims.bind(py).call_method0("tolist")?),
                repr_string(&expected_keepdims.call_method0("tolist")?)
            );

            let zero_scalar = numeric_array(py, 0_i64, "int64");
            let zero_actual = count_nonzero(py, zero_scalar.clone().unbind(), None, false)?;
            let zero_expected = numpy.call_method1("count_nonzero", (zero_scalar,))?;
            assert_eq!(
                repr_string(zero_actual.bind(py)),
                repr_string(&zero_expected)
            );

            let nonzero_scalar = numeric_array(py, 7_i64, "int64");
            let nonzero_actual = count_nonzero(py, nonzero_scalar.clone().unbind(), None, false)?;
            let nonzero_expected = numpy.call_method1("count_nonzero", (nonzero_scalar,))?;
            assert_eq!(
                repr_string(nonzero_actual.bind(py)),
                repr_string(&nonzero_expected)
            );

            let false_scalar = numeric_array(py, false, "bool");
            let false_actual = count_nonzero(py, false_scalar.clone().unbind(), None, false)?;
            let false_expected = numpy.call_method1("count_nonzero", (false_scalar,))?;
            assert_eq!(
                repr_string(false_actual.bind(py)),
                repr_string(&false_expected)
            );

            let true_scalar = numeric_array(py, true, "bool");
            let true_actual = count_nonzero(py, true_scalar.clone().unbind(), None, false)?;
            let true_expected = numpy.call_method1("count_nonzero", (true_scalar,))?;
            assert_eq!(
                repr_string(true_actual.bind(py)),
                repr_string(&true_expected)
            );

            Ok(())
        });
    }

    #[test]
    fn slogdet_matches_numpy_namedtuple_across_real_complex_and_batched() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let slogdet_fn = module.getattr("slogdet")?;
            let numpy = py.import("numpy")?;
            let numpy_slogdet = numpy.getattr("linalg")?.getattr("slogdet")?;
            let isclose = numpy.getattr("isclose")?;
            let allclose = numpy.getattr("allclose")?;

            // Positive-determinant real 2x2.
            let a_pos = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [3.0_f64, 1.0])?,
                    PyList::new(py, [1.0_f64, 2.0])?,
                ],
            )?,))?;
            let actual = slogdet_fn.call1((a_pos.clone(),))?;
            let expected = numpy_slogdet.call1((a_pos.clone(),))?;
            let actual_sign = actual.getattr("sign")?;
            let expected_sign = expected.getattr("sign")?;
            let actual_logabsdet = actual.getattr("logabsdet")?;
            let expected_logabsdet = expected.getattr("logabsdet")?;
            assert!(
                isclose
                    .call1((&actual_sign, &expected_sign))?
                    .extract::<bool>()?,
                "slogdet sign (pos) diverged"
            );
            assert!(
                isclose
                    .call1((&actual_logabsdet, &expected_logabsdet))?
                    .extract::<bool>()?,
                "slogdet logabsdet (pos) diverged"
            );

            // Negative-determinant real 2x2 (sign = -1).
            let a_neg = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [0.0_f64, 1.0])?,
                    PyList::new(py, [1.0_f64, 0.0])?,
                ],
            )?,))?;
            let actual_n = slogdet_fn.call1((a_neg.clone(),))?;
            let expected_n = numpy_slogdet.call1((a_neg.clone(),))?;
            assert!(
                isclose
                    .call1((&actual_n.getattr("sign")?, &expected_n.getattr("sign")?))?
                    .extract::<bool>()?,
                "slogdet sign (neg) diverged"
            );

            // Singular real 2x2 — sign=0, logabsdet=-inf.
            let a_sing = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1.0_f64, 2.0])?,
                    PyList::new(py, [2.0_f64, 4.0])?,
                ],
            )?,))?;
            let actual_s = slogdet_fn.call1((a_sing.clone(),))?;
            let expected_s = numpy_slogdet.call1((a_sing.clone(),))?;
            let a_sign_s = actual_s.getattr("sign")?.extract::<f64>()?;
            let e_sign_s = expected_s.getattr("sign")?.extract::<f64>()?;
            assert_eq!(a_sign_s, e_sign_s, "slogdet singular sign diverged");

            // Batched stack of three 2x2 matrices.
            let batched = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(
                        py,
                        [
                            PyList::new(py, [2.0_f64, 0.0])?,
                            PyList::new(py, [0.0_f64, 3.0])?,
                        ],
                    )?,
                    PyList::new(
                        py,
                        [
                            PyList::new(py, [1.0_f64, 0.0])?,
                            PyList::new(py, [0.0_f64, -1.0])?,
                        ],
                    )?,
                    PyList::new(
                        py,
                        [
                            PyList::new(py, [4.0_f64, 1.0])?,
                            PyList::new(py, [2.0_f64, 3.0])?,
                        ],
                    )?,
                ],
            )?,))?;
            let actual_b = slogdet_fn.call1((batched.clone(),))?;
            let expected_b = numpy_slogdet.call1((batched.clone(),))?;
            assert!(
                allclose
                    .call1((&actual_b.getattr("sign")?, &expected_b.getattr("sign")?))?
                    .extract::<bool>()?,
                "slogdet batched sign diverged"
            );
            assert!(
                allclose
                    .call1((
                        &actual_b.getattr("logabsdet")?,
                        &expected_b.getattr("logabsdet")?,
                    ))?
                    .extract::<bool>()?,
                "slogdet batched logabsdet diverged"
            );

            // Complex 2x2.
            let builtins = py.import("builtins")?;
            let complex_a = numpy.getattr("array")?.call(
                (PyList::new(
                    py,
                    [
                        PyList::new(
                            py,
                            [
                                builtins.getattr("complex")?.call1((1.0_f64, 1.0_f64))?,
                                builtins.getattr("complex")?.call1((2.0_f64, 0.0_f64))?,
                            ],
                        )?,
                        PyList::new(
                            py,
                            [
                                builtins.getattr("complex")?.call1((0.0_f64, -1.0_f64))?,
                                builtins.getattr("complex")?.call1((1.0_f64, 2.0_f64))?,
                            ],
                        )?,
                    ],
                )?,),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("dtype", "complex128")?;
                    kwargs
                }),
            )?;
            let actual_c = slogdet_fn.call1((complex_a.clone(),))?;
            let expected_c = numpy_slogdet.call1((complex_a.clone(),))?;
            assert!(
                isclose
                    .call1((&actual_c.getattr("sign")?, &expected_c.getattr("sign")?))?
                    .extract::<bool>()?,
                "slogdet complex sign diverged"
            );
            assert!(
                isclose
                    .call1((
                        &actual_c.getattr("logabsdet")?,
                        &expected_c.getattr("logabsdet")?,
                    ))?
                    .extract::<bool>()?,
                "slogdet complex logabsdet diverged"
            );

            Ok(())
        });
    }

    #[test]
    fn svd_matches_numpy_namedtuple_array_and_error_paths() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let svd_fn = module.getattr("svd")?;
            let numpy = py.import("numpy")?;
            let numpy_svd = numpy.getattr("linalg")?.getattr("svd")?;
            let builtins = py.import("builtins")?;

            let matrix = numeric_array(
                py,
                vec![vec![1.0_f64, 2.0, 3.0], vec![4.0_f64, 5.0, 6.0]],
                "float64",
            );
            let actual_default = svd_fn.call1((matrix.clone(),))?;
            let expected_default = numpy_svd.call1((matrix.clone(),))?;
            assert_array_matches_numpy(
                &actual_default.getattr("U")?,
                &expected_default.getattr("U")?,
            )?;
            assert_array_matches_numpy(
                &actual_default.getattr("S")?,
                &expected_default.getattr("S")?,
            )?;
            assert_array_matches_numpy(
                &actual_default.getattr("Vh")?,
                &expected_default.getattr("Vh")?,
            )?;

            let reduced_kwargs = PyDict::new(py);
            reduced_kwargs.set_item("full_matrices", false)?;
            let actual_reduced = svd_fn.call((matrix.clone(),), Some(&reduced_kwargs))?;
            let expected_reduced = numpy_svd.call((matrix.clone(),), Some(&reduced_kwargs))?;
            assert_array_matches_numpy(
                &actual_reduced.getattr("U")?,
                &expected_reduced.getattr("U")?,
            )?;
            assert_array_matches_numpy(
                &actual_reduced.getattr("S")?,
                &expected_reduced.getattr("S")?,
            )?;
            assert_array_matches_numpy(
                &actual_reduced.getattr("Vh")?,
                &expected_reduced.getattr("Vh")?,
            )?;

            let values_only_kwargs = PyDict::new(py);
            values_only_kwargs.set_item("compute_uv", false)?;
            let actual_values = svd_fn.call((matrix.clone(),), Some(&values_only_kwargs))?;
            let expected_values = numpy_svd.call((matrix.clone(),), Some(&values_only_kwargs))?;
            assert_array_matches_numpy(&actual_values, &expected_values)?;

            let hermitian_matrix = numeric_array(
                py,
                vec![
                    vec![5.0_f64, 2.0, 1.0],
                    vec![2.0_f64, 4.0, 0.0],
                    vec![1.0_f64, 0.0, 3.0],
                ],
                "float64",
            );
            let hermitian_kwargs = PyDict::new(py);
            hermitian_kwargs.set_item("hermitian", true)?;
            let actual_hermitian =
                svd_fn.call((hermitian_matrix.clone(),), Some(&hermitian_kwargs))?;
            let expected_hermitian =
                numpy_svd.call((hermitian_matrix.clone(),), Some(&hermitian_kwargs))?;
            assert_array_matches_numpy(
                &actual_hermitian.getattr("U")?,
                &expected_hermitian.getattr("U")?,
            )?;
            assert_array_matches_numpy(
                &actual_hermitian.getattr("S")?,
                &expected_hermitian.getattr("S")?,
            )?;
            assert_array_matches_numpy(
                &actual_hermitian.getattr("Vh")?,
                &expected_hermitian.getattr("Vh")?,
            )?;

            let complex_batch = numpy.getattr("array")?.call(
                (PyList::new(
                    py,
                    [
                        PyList::new(
                            py,
                            [
                                PyList::new(
                                    py,
                                    [
                                        builtins.getattr("complex")?.call1((1.0_f64, 1.0_f64))?,
                                        builtins.getattr("complex")?.call1((2.0_f64, -1.0_f64))?,
                                    ],
                                )?,
                                PyList::new(
                                    py,
                                    [
                                        builtins.getattr("complex")?.call1((0.0_f64, 2.0_f64))?,
                                        builtins.getattr("complex")?.call1((3.0_f64, 0.5_f64))?,
                                    ],
                                )?,
                            ],
                        )?,
                        PyList::new(
                            py,
                            [
                                PyList::new(
                                    py,
                                    [
                                        builtins.getattr("complex")?.call1((2.0_f64, 0.0_f64))?,
                                        builtins.getattr("complex")?.call1((1.0_f64, 1.0_f64))?,
                                    ],
                                )?,
                                PyList::new(
                                    py,
                                    [
                                        builtins.getattr("complex")?.call1((4.0_f64, -1.0_f64))?,
                                        builtins.getattr("complex")?.call1((0.5_f64, 2.0_f64))?,
                                    ],
                                )?,
                            ],
                        )?,
                    ],
                )?,),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("dtype", "complex128")?;
                    kwargs
                }),
            )?;
            let actual_batch = svd_fn.call1((complex_batch.clone(),))?;
            let expected_batch = numpy_svd.call1((complex_batch.clone(),))?;
            assert_array_matches_numpy(&actual_batch.getattr("U")?, &expected_batch.getattr("U")?)?;
            assert_array_matches_numpy(&actual_batch.getattr("S")?, &expected_batch.getattr("S")?)?;
            assert_array_matches_numpy(
                &actual_batch.getattr("Vh")?,
                &expected_batch.getattr("Vh")?,
            )?;

            let vector = numeric_array(py, vec![1.0_f64, 2.0, 3.0], "float64");
            let actual_error =
                call_outcome(py, &svd_fn, &PyTuple::new(py, [vector.clone()])?, None)?;
            let expected_error = call_outcome(py, &numpy_svd, &PyTuple::new(py, [vector])?, None)?;
            assert_eq!(actual_error, expected_error);

            Ok(())
        });
    }

    #[test]
    fn qr_matches_numpy_modes_and_error_surface() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let qr_fn = module.getattr("qr")?;
            let numpy = py.import("numpy")?;
            let numpy_qr = numpy.getattr("linalg")?.getattr("qr")?;
            let builtins = py.import("builtins")?;

            let tall = numeric_array(
                py,
                vec![vec![1.0_f64, 2.0], vec![3.0_f64, 4.0], vec![5.0_f64, 6.0]],
                "float64",
            );
            let actual_reduced = qr_fn.call1((tall.clone(),))?;
            let expected_reduced = numpy_qr.call1((tall.clone(),))?;
            assert_array_matches_numpy(
                &actual_reduced.getattr("Q")?,
                &expected_reduced.getattr("Q")?,
            )?;
            assert_array_matches_numpy(
                &actual_reduced.getattr("R")?,
                &expected_reduced.getattr("R")?,
            )?;

            let complete_kwargs = PyDict::new(py);
            complete_kwargs.set_item("mode", "complete")?;
            let actual_complete = qr_fn.call((tall.clone(),), Some(&complete_kwargs))?;
            let expected_complete = numpy_qr.call((tall.clone(),), Some(&complete_kwargs))?;
            assert_array_matches_numpy(
                &actual_complete.getattr("Q")?,
                &expected_complete.getattr("Q")?,
            )?;
            assert_array_matches_numpy(
                &actual_complete.getattr("R")?,
                &expected_complete.getattr("R")?,
            )?;

            let r_kwargs = PyDict::new(py);
            r_kwargs.set_item("mode", "r")?;
            let actual_r = qr_fn.call((tall.clone(),), Some(&r_kwargs))?;
            let expected_r = numpy_qr.call((tall.clone(),), Some(&r_kwargs))?;
            assert_array_matches_numpy(&actual_r, &expected_r)?;

            let raw_kwargs = PyDict::new(py);
            raw_kwargs.set_item("mode", "raw")?;
            let actual_raw = qr_fn.call((tall.clone(),), Some(&raw_kwargs))?;
            let expected_raw = numpy_qr.call((tall.clone(),), Some(&raw_kwargs))?;
            assert_index_tuple_matches_numpy(&actual_raw, &expected_raw)?;

            let full_outcome = call_outcome(
                py,
                &qr_fn,
                &PyTuple::new(py, [tall.clone()])?,
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("mode", "full")?;
                    kwargs
                }),
            )?;
            let expected_full_outcome = call_outcome(
                py,
                &numpy_qr,
                &PyTuple::new(py, [tall.clone()])?,
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("mode", "full")?;
                    kwargs
                }),
            )?;
            assert_eq!(full_outcome, expected_full_outcome);

            let economic_outcome = call_outcome(
                py,
                &qr_fn,
                &PyTuple::new(py, [tall.clone()])?,
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("mode", "economic")?;
                    kwargs
                }),
            )?;
            let expected_economic_outcome = call_outcome(
                py,
                &numpy_qr,
                &PyTuple::new(py, [tall.clone()])?,
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("mode", "economic")?;
                    kwargs
                }),
            )?;
            assert_eq!(economic_outcome, expected_economic_outcome);

            let alias_outcome = call_outcome(
                py,
                &qr_fn,
                &PyTuple::new(py, [tall.clone()])?,
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("mode", "f")?;
                    kwargs
                }),
            )?;
            let expected_alias_outcome = call_outcome(
                py,
                &numpy_qr,
                &PyTuple::new(py, [tall.clone()])?,
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("mode", "f")?;
                    kwargs
                }),
            )?;
            assert_eq!(alias_outcome, expected_alias_outcome);

            let complex_batch = numpy.getattr("array")?.call(
                (PyList::new(
                    py,
                    [
                        PyList::new(
                            py,
                            [
                                PyList::new(
                                    py,
                                    [
                                        builtins.getattr("complex")?.call1((1.0_f64, 1.0_f64))?,
                                        builtins.getattr("complex")?.call1((2.0_f64, 0.0_f64))?,
                                    ],
                                )?,
                                PyList::new(
                                    py,
                                    [
                                        builtins.getattr("complex")?.call1((3.0_f64, -1.0_f64))?,
                                        builtins.getattr("complex")?.call1((4.0_f64, 2.0_f64))?,
                                    ],
                                )?,
                            ],
                        )?,
                        PyList::new(
                            py,
                            [
                                PyList::new(
                                    py,
                                    [
                                        builtins.getattr("complex")?.call1((0.5_f64, 0.0_f64))?,
                                        builtins.getattr("complex")?.call1((1.5_f64, -0.5_f64))?,
                                    ],
                                )?,
                                PyList::new(
                                    py,
                                    [
                                        builtins.getattr("complex")?.call1((2.0_f64, 1.0_f64))?,
                                        builtins.getattr("complex")?.call1((1.0_f64, 0.0_f64))?,
                                    ],
                                )?,
                            ],
                        )?,
                    ],
                )?,),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("dtype", "complex128")?;
                    kwargs
                }),
            )?;
            let actual_batch = qr_fn.call1((complex_batch.clone(),))?;
            let expected_batch = numpy_qr.call1((complex_batch.clone(),))?;
            assert_array_matches_numpy(&actual_batch.getattr("Q")?, &expected_batch.getattr("Q")?)?;
            assert_array_matches_numpy(&actual_batch.getattr("R")?, &expected_batch.getattr("R")?)?;

            let bad_mode = call_outcome(
                py,
                &qr_fn,
                &PyTuple::new(py, [tall.clone()])?,
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("mode", "hostile_mode")?;
                    kwargs
                }),
            )?;
            let expected_bad_mode = call_outcome(
                py,
                &numpy_qr,
                &PyTuple::new(py, [tall])?,
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("mode", "hostile_mode")?;
                    kwargs
                }),
            )?;
            assert_eq!(bad_mode, expected_bad_mode);

            Ok(())
        });
    }

    #[test]
    fn cholesky_matches_numpy_lower_upper_complex_and_error_surface() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let cholesky_fn = module.getattr("cholesky")?;
            let numpy = py.import("numpy")?;
            let numpy_cholesky = numpy.getattr("linalg")?.getattr("cholesky")?;
            let builtins = py.import("builtins")?;

            let spd = numeric_array(
                py,
                vec![
                    vec![4.0_f64, 1.0, 2.0],
                    vec![1.0_f64, 3.0, 0.0],
                    vec![2.0_f64, 0.0, 5.0],
                ],
                "float64",
            );
            let actual_lower = cholesky_fn.call1((spd.clone(),))?;
            let expected_lower = numpy_cholesky.call1((spd.clone(),))?;
            assert_array_matches_numpy(&actual_lower, &expected_lower)?;

            let upper_kwargs = PyDict::new(py);
            upper_kwargs.set_item("upper", true)?;
            let actual_upper = cholesky_fn.call((spd.clone(),), Some(&upper_kwargs))?;
            let expected_upper = numpy_cholesky.call((spd.clone(),), Some(&upper_kwargs))?;
            assert_array_matches_numpy(&actual_upper, &expected_upper)?;

            let batched = numeric_array(
                py,
                vec![
                    vec![vec![4.0_f64, 1.0], vec![1.0_f64, 3.0]],
                    vec![vec![9.0_f64, 0.0], vec![0.0_f64, 16.0]],
                ],
                "float64",
            );
            let actual_batch = cholesky_fn.call1((batched.clone(),))?;
            let expected_batch = numpy_cholesky.call1((batched.clone(),))?;
            assert_array_matches_numpy(&actual_batch, &expected_batch)?;

            let complex_spd = numpy.getattr("array")?.call(
                (PyList::new(
                    py,
                    [
                        PyList::new(
                            py,
                            [
                                builtins.getattr("complex")?.call1((5.0_f64, 0.0_f64))?,
                                builtins.getattr("complex")?.call1((1.0_f64, -2.0_f64))?,
                            ],
                        )?,
                        PyList::new(
                            py,
                            [
                                builtins.getattr("complex")?.call1((1.0_f64, 2.0_f64))?,
                                builtins.getattr("complex")?.call1((6.0_f64, 0.0_f64))?,
                            ],
                        )?,
                    ],
                )?,),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("dtype", "complex128")?;
                    kwargs
                }),
            )?;
            let actual_complex = cholesky_fn.call1((complex_spd.clone(),))?;
            let expected_complex = numpy_cholesky.call1((complex_spd.clone(),))?;
            assert_array_matches_numpy(&actual_complex, &expected_complex)?;

            let actual_positional_err = cholesky_fn.call1((spd.clone(), true)).unwrap_err();
            let expected_positional_err = numpy_cholesky.call1((spd.clone(), true)).unwrap_err();
            assert_eq!(
                actual_positional_err
                    .get_type(py)
                    .name()?
                    .extract::<String>()?,
                expected_positional_err
                    .get_type(py)
                    .name()?
                    .extract::<String>()?
            );
            assert_eq!(
                actual_positional_err.value(py).str()?.extract::<String>()?,
                expected_positional_err
                    .value(py)
                    .str()?
                    .extract::<String>()?
            );

            let non_pd = numeric_array(py, vec![vec![1.0_f64, 2.0], vec![2.0_f64, 1.0]], "float64");
            let actual_error =
                call_outcome(py, &cholesky_fn, &PyTuple::new(py, [non_pd.clone()])?, None)?;
            let expected_error =
                call_outcome(py, &numpy_cholesky, &PyTuple::new(py, [non_pd])?, None)?;
            assert_eq!(actual_error, expected_error);

            Ok(())
        });
    }

    #[test]
    fn matrix_rank_matches_numpy_across_tol_rtol_hermitian_and_batched() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let matrix_rank_fn = module.getattr("matrix_rank")?;
            let numpy = py.import("numpy")?;
            let numpy_matrix_rank = numpy.getattr("linalg")?.getattr("matrix_rank")?;
            let array_equal = numpy.getattr("array_equal")?;

            let assert_scalar_eq =
                |actual: &Bound<'_, PyAny>, expected: &Bound<'_, PyAny>| -> PyResult<()> {
                    let a_val = actual.extract::<i64>()?;
                    let e_val = expected.extract::<i64>()?;
                    assert_eq!(a_val, e_val, "matrix_rank scalar diverged");
                    Ok(())
                };

            // Full-rank 3x3.
            let full_rank = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1.0_f64, 0.0, 0.0])?,
                    PyList::new(py, [0.0_f64, 1.0, 0.0])?,
                    PyList::new(py, [0.0_f64, 0.0, 1.0])?,
                ],
            )?,))?;
            assert_scalar_eq(
                &matrix_rank_fn.call1((full_rank.clone(),))?,
                &numpy_matrix_rank.call1((full_rank.clone(),))?,
            )?;

            // Rank-deficient 3x3 (two linearly dependent rows).
            let defic = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1.0_f64, 2.0, 3.0])?,
                    PyList::new(py, [2.0_f64, 4.0, 6.0])?,
                    PyList::new(py, [0.0_f64, 1.0, 2.0])?,
                ],
            )?,))?;
            assert_scalar_eq(
                &matrix_rank_fn.call1((defic.clone(),))?,
                &numpy_matrix_rank.call1((defic.clone(),))?,
            )?;

            // 1-D vector — nonzero yields rank 1.
            let vec_nonzero = numpy.getattr("array")?.call1((vec![1.0_f64, 2.0, 3.0],))?;
            assert_scalar_eq(
                &matrix_rank_fn.call1((vec_nonzero.clone(),))?,
                &numpy_matrix_rank.call1((vec_nonzero.clone(),))?,
            )?;

            // Explicit tol overrides auto-threshold.
            let tol_kwargs = PyDict::new(py);
            tol_kwargs.set_item("tol", 0.5_f64)?;
            let actual_tol = matrix_rank_fn.call((defic.clone(),), Some(&tol_kwargs))?;
            let tol_kwargs_n = PyDict::new(py);
            tol_kwargs_n.set_item("tol", 0.5_f64)?;
            let expected_tol = numpy_matrix_rank.call((defic.clone(),), Some(&tol_kwargs_n))?;
            assert_scalar_eq(&actual_tol, &expected_tol)?;

            // rtol keyword-only on the same matrix.
            let rtol_kwargs = PyDict::new(py);
            rtol_kwargs.set_item("rtol", 1e-3_f64)?;
            let actual_rtol = matrix_rank_fn.call((defic.clone(),), Some(&rtol_kwargs))?;
            let rtol_kwargs_n = PyDict::new(py);
            rtol_kwargs_n.set_item("rtol", 1e-3_f64)?;
            let expected_rtol = numpy_matrix_rank.call((defic.clone(),), Some(&rtol_kwargs_n))?;
            assert_scalar_eq(&actual_rtol, &expected_rtol)?;

            // hermitian=True on a symmetric SPD 2x2.
            let spd = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [2.0_f64, 1.0])?,
                    PyList::new(py, [1.0_f64, 2.0])?,
                ],
            )?,))?;
            let herm_kwargs = PyDict::new(py);
            herm_kwargs.set_item("hermitian", true)?;
            let actual_h = matrix_rank_fn.call((spd.clone(),), Some(&herm_kwargs))?;
            let herm_kwargs_n = PyDict::new(py);
            herm_kwargs_n.set_item("hermitian", true)?;
            let expected_h = numpy_matrix_rank.call((spd.clone(),), Some(&herm_kwargs_n))?;
            assert_scalar_eq(&actual_h, &expected_h)?;

            // Batched stack of three 2x2 matrices (full-rank, rank-1, zero).
            let batched = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(
                        py,
                        [
                            PyList::new(py, [1.0_f64, 0.0])?,
                            PyList::new(py, [0.0_f64, 1.0])?,
                        ],
                    )?,
                    PyList::new(
                        py,
                        [
                            PyList::new(py, [1.0_f64, 2.0])?,
                            PyList::new(py, [2.0_f64, 4.0])?,
                        ],
                    )?,
                    PyList::new(
                        py,
                        [
                            PyList::new(py, [0.0_f64, 0.0])?,
                            PyList::new(py, [0.0_f64, 0.0])?,
                        ],
                    )?,
                ],
            )?,))?;
            let actual_b = matrix_rank_fn.call1((batched.clone(),))?;
            let expected_b = numpy_matrix_rank.call1((batched.clone(),))?;
            assert!(
                array_equal
                    .call1((&actual_b, &expected_b))?
                    .extract::<bool>()?,
                "matrix_rank batched diverged"
            );

            Ok(())
        });
    }

    #[test]
    fn matrix_power_matches_numpy_across_dtypes_negative_and_stacked_inputs() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let matrix_power_fn = module.getattr("matrix_power")?;
            let numpy = py.import("numpy")?;
            let numpy_matrix_power = numpy.getattr("linalg")?.getattr("matrix_power")?;

            let int_matrix =
                numeric_array(py, vec![vec![1_i64, 2_i64], vec![3_i64, 5_i64]], "int64");
            let actual_int = matrix_power_fn.call1((int_matrix.clone(), 3_i64))?;
            let expected_int = numpy_matrix_power.call1((int_matrix.clone(), 3_i64))?;
            assert_array_matches_numpy(&actual_int, &expected_int)?;

            let bool_matrix = numeric_array(py, vec![vec![true, false], vec![false, true]], "bool");
            let actual_zero = matrix_power_fn.call1((bool_matrix.clone(), 0_i64))?;
            let expected_zero = numpy_matrix_power.call1((bool_matrix.clone(), 0_i64))?;
            assert_array_matches_numpy(&actual_zero, &expected_zero)?;

            let float32_matrix = numeric_array(
                py,
                vec![vec![1.0_f32, 2.0_f32], vec![3.0_f32, 5.0_f32]],
                "float32",
            );
            let actual_neg = matrix_power_fn.call1((float32_matrix.clone(), -1_i64))?;
            let expected_neg = numpy_matrix_power.call1((float32_matrix.clone(), -1_i64))?;
            assert_array_matches_numpy(&actual_neg, &expected_neg)?;

            let stacked = numpy
                .getattr("arange")?
                .call1((8.0_f64,))?
                .call_method1("reshape", ((2, 2, 2),))?;
            let actual_stacked = matrix_power_fn.call1((stacked.clone(), 2_i64))?;
            let expected_stacked = numpy_matrix_power.call1((stacked.clone(), 2_i64))?;
            assert_array_matches_numpy(&actual_stacked, &expected_stacked)?;

            Ok(())
        });
    }

    #[test]
    fn matrix_power_error_surface_matches_numpy() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let matrix_power_fn = module.getattr("matrix_power")?;
            let numpy = py.import("numpy")?;
            let numpy_matrix_power = numpy.getattr("linalg")?.getattr("matrix_power")?;

            let nonsquare = numeric_array(
                py,
                vec![
                    vec![1.0_f64, 2.0_f64, 3.0_f64],
                    vec![4.0_f64, 5.0_f64, 6.0_f64],
                ],
                "float64",
            );
            let actual_nonsquare = matrix_power_fn
                .call1((nonsquare.clone(), 2_i64))
                .unwrap_err();
            let expected_nonsquare = numpy_matrix_power
                .call1((nonsquare.clone(), 2_i64))
                .unwrap_err();
            assert_eq!(
                actual_nonsquare.get_type(py).name()?.extract::<String>()?,
                expected_nonsquare
                    .get_type(py)
                    .name()?
                    .extract::<String>()?
            );
            assert_eq!(
                actual_nonsquare.value(py).str()?.extract::<String>()?,
                expected_nonsquare.value(py).str()?.extract::<String>()?
            );

            let square = numeric_array(
                py,
                vec![vec![1.0_f64, 2.0_f64], vec![3.0_f64, 4.0_f64]],
                "float64",
            );
            let actual_float_exp = matrix_power_fn
                .call1((square.clone(), 1.5_f64))
                .unwrap_err();
            let expected_float_exp = numpy_matrix_power
                .call1((square.clone(), 1.5_f64))
                .unwrap_err();
            assert_eq!(
                actual_float_exp.get_type(py).name()?.extract::<String>()?,
                expected_float_exp
                    .get_type(py)
                    .name()?
                    .extract::<String>()?
            );
            assert_eq!(
                actual_float_exp.value(py).str()?.extract::<String>()?,
                expected_float_exp.value(py).str()?.extract::<String>()?
            );

            Ok(())
        });
    }

    #[test]
    fn solve_matches_numpy_across_square_batched_multi_rhs_and_complex() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let solve_fn = module.getattr("solve")?;
            let numpy = py.import("numpy")?;
            let numpy_solve = numpy.getattr("linalg")?.getattr("solve")?;
            let allclose = numpy.getattr("allclose")?;

            // Square 2x2 with 1-D b.
            let square = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [3.0_f64, 1.0])?,
                    PyList::new(py, [1.0_f64, 2.0])?,
                ],
            )?,))?;
            let b1 = numpy.getattr("array")?.call1((vec![9.0_f64, 8.0],))?;
            let actual = solve_fn.call1((square.clone(), b1.clone()))?;
            let expected = numpy_solve.call1((square.clone(), b1.clone()))?;
            assert!(
                allclose.call1((&actual, &expected))?.extract::<bool>()?,
                "solve 2x2 1-D diverged"
            );

            // 3x3 with 2-D b (multiple RHS).
            let three = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1.0_f64, 2.0, 0.0])?,
                    PyList::new(py, [0.0_f64, 1.0, 3.0])?,
                    PyList::new(py, [4.0_f64, 0.0, 1.0])?,
                ],
            )?,))?;
            let multi_b = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1.0_f64, 10.0])?,
                    PyList::new(py, [2.0_f64, 20.0])?,
                    PyList::new(py, [3.0_f64, 30.0])?,
                ],
            )?,))?;
            let actual_multi = solve_fn.call1((three.clone(), multi_b.clone()))?;
            let expected_multi = numpy_solve.call1((three.clone(), multi_b.clone()))?;
            assert!(
                allclose
                    .call1((&actual_multi, &expected_multi))?
                    .extract::<bool>()?,
                "solve 3x3 multi-RHS diverged"
            );

            // Identity — solve(I, b) == b.
            let eye = numpy.getattr("eye")?.call1((4_i64,))?;
            let b4 = numpy
                .getattr("array")?
                .call1((vec![1.0_f64, -2.0, 3.5, 0.0],))?;
            let actual_eye = solve_fn.call1((eye.clone(), b4.clone()))?;
            let expected_eye = numpy_solve.call1((eye.clone(), b4.clone()))?;
            assert!(
                allclose
                    .call1((&actual_eye, &expected_eye))?
                    .extract::<bool>()?,
                "solve(eye, b) diverged"
            );

            // Batched stack of two 2x2 solves with 2-D b per slice.
            let batched_a = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(
                        py,
                        [
                            PyList::new(py, [2.0_f64, 0.0])?,
                            PyList::new(py, [0.0_f64, 2.0])?,
                        ],
                    )?,
                    PyList::new(
                        py,
                        [
                            PyList::new(py, [1.0_f64, 1.0])?,
                            PyList::new(py, [0.0_f64, 1.0])?,
                        ],
                    )?,
                ],
            )?,))?;
            let batched_b = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [2.0_f64, 4.0])?,
                    PyList::new(py, [1.0_f64, 3.0])?,
                ],
            )?,))?;
            let actual_batch = solve_fn.call1((batched_a.clone(), batched_b.clone()))?;
            let expected_batch = numpy_solve.call1((batched_a.clone(), batched_b.clone()))?;
            assert!(
                allclose
                    .call1((&actual_batch, &expected_batch))?
                    .extract::<bool>()?,
                "solve batched diverged"
            );

            // Complex 2x2 solve.
            let builtins = py.import("builtins")?;
            let complex_a = numpy.getattr("array")?.call(
                (PyList::new(
                    py,
                    [
                        PyList::new(
                            py,
                            [
                                builtins.getattr("complex")?.call1((1.0_f64, 1.0_f64))?,
                                builtins.getattr("complex")?.call1((2.0_f64, 0.0_f64))?,
                            ],
                        )?,
                        PyList::new(
                            py,
                            [
                                builtins.getattr("complex")?.call1((0.0_f64, -1.0_f64))?,
                                builtins.getattr("complex")?.call1((1.0_f64, 2.0_f64))?,
                            ],
                        )?,
                    ],
                )?,),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("dtype", "complex128")?;
                    kwargs
                }),
            )?;
            let complex_b = numpy.getattr("array")?.call(
                (PyList::new(
                    py,
                    [
                        builtins.getattr("complex")?.call1((3.0_f64, 0.0_f64))?,
                        builtins.getattr("complex")?.call1((1.0_f64, 1.0_f64))?,
                    ],
                )?,),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("dtype", "complex128")?;
                    kwargs
                }),
            )?;
            let actual_c = solve_fn.call1((complex_a.clone(), complex_b.clone()))?;
            let expected_c = numpy_solve.call1((complex_a.clone(), complex_b.clone()))?;
            assert!(
                allclose
                    .call1((&actual_c, &expected_c))?
                    .extract::<bool>()?,
                "solve complex diverged"
            );

            Ok(())
        });
    }

    #[test]
    fn eigvalsh_matches_numpy_across_uplo_batched_and_complex() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let eigvalsh_fn = module.getattr("eigvalsh")?;
            let numpy = py.import("numpy")?;
            let numpy_eigvalsh = numpy.getattr("linalg")?.getattr("eigvalsh")?;
            let allclose = numpy.getattr("allclose")?;

            // Real symmetric 2x2 — default UPLO='L'.
            let spd = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [2.0_f64, 1.0])?,
                    PyList::new(py, [1.0_f64, 2.0])?,
                ],
            )?,))?;
            let actual = eigvalsh_fn.call1((spd.clone(),))?;
            let expected = numpy_eigvalsh.call1((spd.clone(),))?;
            assert!(
                allclose.call1((&actual, &expected))?.extract::<bool>()?,
                "eigvalsh 2x2 default UPLO diverged"
            );

            // Explicit UPLO='U' on a 3x3 symmetric — both triangles match by construction.
            let sym3 = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [4.0_f64, 1.0, 0.0])?,
                    PyList::new(py, [1.0_f64, 3.0, 2.0])?,
                    PyList::new(py, [0.0_f64, 2.0, 5.0])?,
                ],
            )?,))?;
            let kwargs_u = PyDict::new(py);
            kwargs_u.set_item("UPLO", "U")?;
            let actual_u = eigvalsh_fn.call((sym3.clone(),), Some(&kwargs_u))?;
            let kwargs_u_n = PyDict::new(py);
            kwargs_u_n.set_item("UPLO", "U")?;
            let expected_u = numpy_eigvalsh.call((sym3.clone(),), Some(&kwargs_u_n))?;
            assert!(
                allclose
                    .call1((&actual_u, &expected_u))?
                    .extract::<bool>()?,
                "eigvalsh UPLO='U' diverged"
            );

            // Identity → eigenvalues all 1.
            let eye = numpy.getattr("eye")?.call1((4_i64,))?;
            let actual_eye = eigvalsh_fn.call1((eye.clone(),))?;
            let expected_eye = numpy_eigvalsh.call1((eye.clone(),))?;
            assert!(
                allclose
                    .call1((&actual_eye, &expected_eye))?
                    .extract::<bool>()?,
                "eigvalsh(eye) diverged"
            );

            // Batched stack of two 2x2 symmetric matrices.
            let batched = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(
                        py,
                        [
                            PyList::new(py, [2.0_f64, 0.0])?,
                            PyList::new(py, [0.0_f64, 3.0])?,
                        ],
                    )?,
                    PyList::new(
                        py,
                        [
                            PyList::new(py, [5.0_f64, 1.0])?,
                            PyList::new(py, [1.0_f64, 4.0])?,
                        ],
                    )?,
                ],
            )?,))?;
            let actual_batch = eigvalsh_fn.call1((batched.clone(),))?;
            let expected_batch = numpy_eigvalsh.call1((batched.clone(),))?;
            assert!(
                allclose
                    .call1((&actual_batch, &expected_batch))?
                    .extract::<bool>()?,
                "eigvalsh batched diverged"
            );

            // Complex Hermitian 2x2 — passthrough must forward numpy's real-eigenvalue guarantee.
            let builtins = py.import("builtins")?;
            let hermitian = numpy.getattr("array")?.call(
                (PyList::new(
                    py,
                    [
                        PyList::new(
                            py,
                            [
                                builtins.getattr("complex")?.call1((2.0_f64, 0.0_f64))?,
                                builtins.getattr("complex")?.call1((1.0_f64, -1.0_f64))?,
                            ],
                        )?,
                        PyList::new(
                            py,
                            [
                                builtins.getattr("complex")?.call1((1.0_f64, 1.0_f64))?,
                                builtins.getattr("complex")?.call1((3.0_f64, 0.0_f64))?,
                            ],
                        )?,
                    ],
                )?,),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("dtype", "complex128")?;
                    kwargs
                }),
            )?;
            let actual_h = eigvalsh_fn.call1((hermitian.clone(),))?;
            let expected_h = numpy_eigvalsh.call1((hermitian.clone(),))?;
            assert!(
                allclose
                    .call1((&actual_h, &expected_h))?
                    .extract::<bool>()?,
                "eigvalsh Hermitian diverged"
            );

            Ok(())
        });
    }

    #[test]
    fn det_and_inv_match_numpy_across_real_complex_and_batched_inputs() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let det_fn = module.getattr("det")?;
            let inv_fn = module.getattr("inv")?;
            let numpy = py.import("numpy")?;
            let numpy_det = numpy.getattr("linalg")?.getattr("det")?;
            let numpy_inv = numpy.getattr("linalg")?.getattr("inv")?;
            let allclose = numpy.getattr("allclose")?;
            let isclose = numpy.getattr("isclose")?;

            // Real 2-D 2x2 invertible — fast path.
            let square = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [4.0_f64, 7.0])?,
                    PyList::new(py, [2.0_f64, 6.0])?,
                ],
            )?,))?;
            let actual_det = det_fn.call1((square.clone(),))?;
            let expected_det = numpy_det.call1((square.clone(),))?;
            assert!(
                isclose
                    .call1((&actual_det, &expected_det))?
                    .extract::<bool>()?,
                "det 2x2 diverged"
            );
            let actual_inv = inv_fn.call1((square.clone(),))?;
            let expected_inv = numpy_inv.call1((square.clone(),))?;
            assert!(
                allclose
                    .call1((&actual_inv, &expected_inv))?
                    .extract::<bool>()?,
                "inv 2x2 diverged"
            );

            // Real 2-D 3x3 invertible.
            let three = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1.0_f64, 2.0, 0.0])?,
                    PyList::new(py, [0.0_f64, 1.0, 3.0])?,
                    PyList::new(py, [4.0_f64, 0.0, 1.0])?,
                ],
            )?,))?;
            let actual_det3 = det_fn.call1((three.clone(),))?;
            let expected_det3 = numpy_det.call1((three.clone(),))?;
            assert!(
                isclose
                    .call1((&actual_det3, &expected_det3))?
                    .extract::<bool>()?,
                "det 3x3 diverged"
            );
            let actual_inv3 = inv_fn.call1((three.clone(),))?;
            let expected_inv3 = numpy_inv.call1((three.clone(),))?;
            assert!(
                allclose
                    .call1((&actual_inv3, &expected_inv3))?
                    .extract::<bool>()?,
                "inv 3x3 diverged"
            );

            // Identity matrix — det(I) = 1, inv(I) = I.
            let eye = numpy.getattr("eye")?.call1((4_i64,))?;
            let actual_det_i = det_fn.call1((eye.clone(),))?;
            let expected_det_i = numpy_det.call1((eye.clone(),))?;
            assert!(
                isclose
                    .call1((&actual_det_i, &expected_det_i))?
                    .extract::<bool>()?,
                "det(eye) diverged"
            );
            let actual_inv_i = inv_fn.call1((eye.clone(),))?;
            let expected_inv_i = numpy_inv.call1((eye.clone(),))?;
            assert!(
                allclose
                    .call1((&actual_inv_i, &expected_inv_i))?
                    .extract::<bool>()?,
                "inv(eye) diverged"
            );

            // Batched 2x2 stack of 3 — passthrough path.
            let batched = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(
                        py,
                        [
                            PyList::new(py, [1.0_f64, 0.0])?,
                            PyList::new(py, [0.0_f64, 1.0])?,
                        ],
                    )?,
                    PyList::new(
                        py,
                        [
                            PyList::new(py, [2.0_f64, 0.0])?,
                            PyList::new(py, [0.0_f64, 2.0])?,
                        ],
                    )?,
                    PyList::new(
                        py,
                        [
                            PyList::new(py, [1.0_f64, 1.0])?,
                            PyList::new(py, [0.0_f64, 1.0])?,
                        ],
                    )?,
                ],
            )?,))?;
            let actual_det_batch = det_fn.call1((batched.clone(),))?;
            let expected_det_batch = numpy_det.call1((batched.clone(),))?;
            assert!(
                allclose
                    .call1((&actual_det_batch, &expected_det_batch))?
                    .extract::<bool>()?,
                "det batched diverged"
            );
            let actual_inv_batch = inv_fn.call1((batched.clone(),))?;
            let expected_inv_batch = numpy_inv.call1((batched.clone(),))?;
            assert!(
                allclose
                    .call1((&actual_inv_batch, &expected_inv_batch))?
                    .extract::<bool>()?,
                "inv batched diverged"
            );

            // Complex 2x2 — passthrough path.
            let builtins = py.import("builtins")?;
            let complex_2x2 = numpy.getattr("array")?.call(
                (PyList::new(
                    py,
                    [
                        PyList::new(
                            py,
                            [
                                builtins.getattr("complex")?.call1((1.0_f64, 2.0_f64))?,
                                builtins.getattr("complex")?.call1((0.0_f64, 1.0_f64))?,
                            ],
                        )?,
                        PyList::new(
                            py,
                            [
                                builtins.getattr("complex")?.call1((3.0_f64, 0.0_f64))?,
                                builtins.getattr("complex")?.call1((1.0_f64, -1.0_f64))?,
                            ],
                        )?,
                    ],
                )?,),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("dtype", "complex128")?;
                    kwargs
                }),
            )?;
            let actual_det_c = det_fn.call1((complex_2x2.clone(),))?;
            let expected_det_c = numpy_det.call1((complex_2x2.clone(),))?;
            assert!(
                isclose
                    .call1((&actual_det_c, &expected_det_c))?
                    .extract::<bool>()?,
                "det complex diverged"
            );
            let actual_inv_c = inv_fn.call1((complex_2x2.clone(),))?;
            let expected_inv_c = numpy_inv.call1((complex_2x2.clone(),))?;
            assert!(
                allclose
                    .call1((&actual_inv_c, &expected_inv_c))?
                    .extract::<bool>()?,
                "inv complex diverged"
            );

            Ok(())
        });
    }

    #[test]
    fn lstsq_matches_numpy_tuple_return_across_shapes_and_rcond() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let lstsq_fn = module.getattr("lstsq")?;
            let numpy = py.import("numpy")?;
            let numpy_lstsq = numpy.getattr("linalg")?.getattr("lstsq")?;
            let allclose = numpy.getattr("allclose")?;
            let array_equal = numpy.getattr("array_equal")?;

            let tuple_close =
                |actual: &Bound<'_, PyAny>, expected: &Bound<'_, PyAny>| -> PyResult<()> {
                    let actual_tuple = actual.downcast::<PyTuple>()?;
                    let expected_tuple = expected.downcast::<PyTuple>()?;
                    assert_eq!(actual_tuple.len()?, expected_tuple.len()?);
                    // 0: solution, 1: residuals, 2: rank (i32), 3: singular values
                    assert!(
                        allclose
                            .call1((actual_tuple.get_item(0)?, expected_tuple.get_item(0)?))?
                            .extract::<bool>()?,
                        "lstsq solution diverged"
                    );
                    assert!(
                        array_equal
                            .call1((actual_tuple.get_item(1)?, expected_tuple.get_item(1)?))?
                            .extract::<bool>()?
                            || allclose
                                .call1((actual_tuple.get_item(1)?, expected_tuple.get_item(1)?))?
                                .extract::<bool>()?,
                        "lstsq residuals diverged"
                    );
                    assert_eq!(
                        actual_tuple.get_item(2)?.extract::<i64>()?,
                        expected_tuple.get_item(2)?.extract::<i64>()?,
                        "lstsq rank diverged"
                    );
                    assert!(
                        allclose
                            .call1((actual_tuple.get_item(3)?, expected_tuple.get_item(3)?))?
                            .extract::<bool>()?,
                        "lstsq singular values diverged"
                    );
                    Ok(())
                };

            // Overdetermined 3x2 (least-squares fit), b is 1-D.
            let tall_a = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1.0_f64, 0.0])?,
                    PyList::new(py, [0.0_f64, 1.0])?,
                    PyList::new(py, [1.0_f64, 1.0])?,
                ],
            )?,))?;
            let tall_b = numpy.getattr("array")?.call1((vec![1.0_f64, 2.0, 3.0],))?;
            let rcond_kwargs = PyDict::new(py);
            rcond_kwargs.set_item("rcond", py.None())?;
            let actual_tall =
                lstsq_fn.call((tall_a.clone(), tall_b.clone()), Some(&rcond_kwargs))?;
            let rcond_kwargs_2 = PyDict::new(py);
            rcond_kwargs_2.set_item("rcond", py.None())?;
            let expected_tall =
                numpy_lstsq.call((tall_a.clone(), tall_b.clone()), Some(&rcond_kwargs_2))?;
            tuple_close(&actual_tall, &expected_tall)?;

            // Square 2x2 exact solve — residuals should be empty.
            let square_a = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [3.0_f64, 1.0])?,
                    PyList::new(py, [1.0_f64, 2.0])?,
                ],
            )?,))?;
            let square_b = numpy.getattr("array")?.call1((vec![9.0_f64, 8.0],))?;
            let rk3 = PyDict::new(py);
            rk3.set_item("rcond", py.None())?;
            let actual_square = lstsq_fn.call((square_a.clone(), square_b.clone()), Some(&rk3))?;
            let rk4 = PyDict::new(py);
            rk4.set_item("rcond", py.None())?;
            let expected_square =
                numpy_lstsq.call((square_a.clone(), square_b.clone()), Some(&rk4))?;
            tuple_close(&actual_square, &expected_square)?;

            // 2-D b (multiple right-hand-sides).
            let multi_b = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1.0_f64, 10.0])?,
                    PyList::new(py, [2.0_f64, 20.0])?,
                    PyList::new(py, [3.0_f64, 30.0])?,
                ],
            )?,))?;
            let rk5 = PyDict::new(py);
            rk5.set_item("rcond", py.None())?;
            let actual_multi = lstsq_fn.call((tall_a.clone(), multi_b.clone()), Some(&rk5))?;
            let rk6 = PyDict::new(py);
            rk6.set_item("rcond", py.None())?;
            let expected_multi = numpy_lstsq.call((tall_a.clone(), multi_b.clone()), Some(&rk6))?;
            tuple_close(&actual_multi, &expected_multi)?;

            // Explicit numeric rcond.
            let rk7 = PyDict::new(py);
            rk7.set_item("rcond", 1e-8_f64)?;
            let actual_r = lstsq_fn.call((tall_a.clone(), tall_b.clone()), Some(&rk7))?;
            let rk8 = PyDict::new(py);
            rk8.set_item("rcond", 1e-8_f64)?;
            let expected_r = numpy_lstsq.call((tall_a.clone(), tall_b.clone()), Some(&rk8))?;
            tuple_close(&actual_r, &expected_r)?;

            Ok(())
        });
    }

    #[test]
    fn tensorsolve_matches_numpy_for_tensor_output_shape() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let numpy = py.import("numpy")?;

            let a = numpy
                .getattr("eye")?
                .call1((8,))?
                .call_method1("reshape", ((2, 2, 2, 2, 2, 2),))?;
            let b = numeric_array(
                py,
                vec![
                    vec![vec![0.0, 1.0], vec![2.0, 3.0]],
                    vec![vec![4.0, 5.0], vec![6.0, 7.0]],
                ],
                "float64",
            );

            let actual = tensorsolve(py, a.clone().unbind(), b.clone().unbind(), None)?;
            let expected = numpy
                .getattr("linalg")?
                .call_method1("tensorsolve", (a, b))?;

            assert_array_matches_numpy(actual.bind(py), &expected)
        });
    }

    #[test]
    fn tensorsolve_matches_numpy_axes_kwarg() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let numpy = py.import("numpy")?;
            let linalg = numpy.getattr("linalg")?;

            let a = numpy
                .getattr("eye")?
                .call1((24,))?
                .call_method1("reshape", ((3, 4, 2, 3, 4, 2),))?;
            let b = numeric_array(
                py,
                vec![
                    vec![
                        vec![0.0, 1.0],
                        vec![2.0, 3.0],
                        vec![4.0, 5.0],
                        vec![6.0, 7.0],
                    ],
                    vec![
                        vec![8.0, 9.0],
                        vec![10.0, 11.0],
                        vec![12.0, 13.0],
                        vec![14.0, 15.0],
                    ],
                    vec![
                        vec![16.0, 17.0],
                        vec![18.0, 19.0],
                        vec![20.0, 21.0],
                        vec![22.0, 23.0],
                    ],
                ],
                "float64",
            );

            for axes in [
                PyTuple::new(py, [0_isize, 2_isize, 1_isize])?
                    .into_any()
                    .unbind(),
                PyList::new(py, [0_isize])?.into_any().unbind(),
                PyTuple::empty(py).into_any().unbind(),
            ] {
                let actual = tensorsolve(
                    py,
                    a.clone().unbind(),
                    b.clone().unbind(),
                    Some(axes.clone_ref(py)),
                )?;
                let kwargs = PyDict::new(py);
                kwargs.set_item("axes", axes.bind(py))?;
                let expected =
                    linalg.call_method("tensorsolve", (a.clone(), b.clone()), Some(&kwargs))?;
                assert_array_matches_numpy(actual.bind(py), &expected)?;
            }

            Ok(())
        });
    }

    #[test]
    fn tensorsolve_axes_error_surface_matches_numpy() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let numpy = py.import("numpy")?;
            let linalg = numpy.getattr("linalg")?;
            let tensorsolve_fn = module.getattr("tensorsolve")?;
            let numpy_tensorsolve = linalg.getattr("tensorsolve")?;

            let a = numpy
                .getattr("eye")?
                .call1((24,))?
                .call_method1("reshape", ((3, 4, 2, 3, 4, 2),))?;
            let b = numpy
                .getattr("arange")?
                .call1((24.0,))?
                .call_method1("reshape", ((3, 4, 2),))?;

            for axes in [
                0_i32.into_pyobject(py)?.unbind().into_any(),
                PyTuple::new(py, [-1_isize])?.into_any().unbind(),
                PyTuple::new(py, [1.0_f64])?.into_any().unbind(),
            ] {
                let actual_kwargs = PyDict::new(py);
                actual_kwargs.set_item("axes", axes.bind(py))?;
                let expected_kwargs = PyDict::new(py);
                expected_kwargs.set_item("axes", axes.bind(py))?;

                let actual = call_outcome(
                    py,
                    &tensorsolve_fn,
                    &PyTuple::new(py, [a.clone(), b.clone()])?,
                    Some(&actual_kwargs),
                )?;
                let expected = call_outcome(
                    py,
                    &numpy_tensorsolve,
                    &PyTuple::new(py, [a.clone(), b.clone()])?,
                    Some(&expected_kwargs),
                )?;
                assert_eq!(actual, expected);
            }

            Ok(())
        });
    }

    #[test]
    fn tensorinv_matches_numpy_default_and_ind_kwarg() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let numpy = py.import("numpy")?;
            let linalg = numpy.getattr("linalg")?;

            let default_input = numpy
                .getattr("eye")?
                .call1((4,))?
                .call_method1("reshape", ((2, 2, 2, 2),))?;
            let actual_default = tensorinv(py, default_input.clone().unbind(), 2)?;
            let expected_default = linalg.call_method1("tensorinv", (default_input,))?;
            assert_array_matches_numpy(actual_default.bind(py), &expected_default)?;

            let ind_input = numpy
                .getattr("eye")?
                .call1((4,))?
                .call_method1("reshape", ((4, 2, 2),))?;
            let actual_ind = tensorinv(py, ind_input.clone().unbind(), 1)?;
            let expected_ind = linalg.call_method(
                "tensorinv",
                (ind_input,),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("ind", 1)?;
                    kwargs
                }),
            )?;
            assert_array_matches_numpy(actual_ind.bind(py), &expected_ind)
        });
    }

    #[test]
    fn trim_zeros_matches_numpy_front_and_back_modes() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let trim_zeros_fn = module.getattr("trim_zeros")?;
            let numpy_trim_zeros = py.import("numpy")?.getattr("trim_zeros")?;

            let list_input = PyList::new(py, [0_i32, 0, 1, 2, 0, 0])?;
            let actual_default = trim_zeros_fn.call1((list_input.clone(),))?;
            let expected_default = numpy_trim_zeros.call1((list_input.clone(),))?;
            assert_eq!(
                actual_default.get_type().name()?.extract::<String>()?,
                expected_default.get_type().name()?.extract::<String>()?
            );
            assert_eq!(repr_string(&actual_default), repr_string(&expected_default));

            let tuple_input = PyTuple::new(py, [0_i32, 0, 1, 2, 0, 0])?;
            let actual_front = trim_zeros_fn.call1((tuple_input.clone(), "f"))?;
            let expected_front = numpy_trim_zeros.call1((tuple_input.clone(), "f"))?;
            assert_eq!(
                actual_front.get_type().name()?.extract::<String>()?,
                expected_front.get_type().name()?.extract::<String>()?
            );
            assert_eq!(repr_string(&actual_front), repr_string(&expected_front));

            let array_input = numeric_array(py, vec![0_i32, 0, 1, 2, 0, 0], "int64");
            let actual_back = trim_zeros_fn.call1((array_input.clone(), "b"))?;
            let expected_back = numpy_trim_zeros.call1((array_input.clone(), "b"))?;
            assert_array_matches_numpy(&actual_back, &expected_back)?;

            let scalar_input = 0_i32.into_pyobject(py)?.unbind();
            let scalar_args = PyTuple::new(py, [scalar_input.clone_ref(py).into_bound(py)])?;
            assert_eq!(
                call_outcome(py, &trim_zeros_fn, &scalar_args, None)?,
                call_outcome(py, &numpy_trim_zeros, &scalar_args, None)?,
            );

            let actual_err = trim_zeros_fn.call1((list_input.clone(), "x")).unwrap_err();
            let expected_err = numpy_trim_zeros.call1((list_input, "x")).unwrap_err();
            assert_eq!(
                actual_err.get_type(py).name()?.extract::<String>()?,
                expected_err.get_type(py).name()?.extract::<String>()?
            );
            assert_eq!(
                actual_err.value(py).str()?.extract::<String>()?,
                expected_err.value(py).str()?.extract::<String>()?
            );

            Ok(())
        });
    }

    #[test]
    fn masked_invalid_matches_numpy_basic_cases_and_copy_semantics() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let masked_invalid_fn = module.getattr("masked_invalid")?;
            let numpy = py.import("numpy")?;
            let numpy_masked_invalid = numpy.getattr("ma")?.getattr("masked_invalid")?;

            let float_input = numeric_array(py, vec![1.0_f64, f64::NAN, f64::INFINITY], "float64");
            let actual_float = masked_invalid_fn.call1((float_input.clone(),))?;
            let expected_float = numpy_masked_invalid.call1((float_input.clone(),))?;
            assert_eq!(repr_string(&actual_float), repr_string(&expected_float));

            let bool_input = numeric_array(py, vec![true, false, true], "bool");
            let actual_bool = masked_invalid_fn.call1((bool_input.clone(),))?;
            let expected_bool = numpy_masked_invalid.call1((bool_input.clone(),))?;
            assert_eq!(repr_string(&actual_bool), repr_string(&expected_bool));

            let builtins = py.import("builtins")?;
            let complex_input = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    builtins.getattr("complex")?.call1((1.0_f64, 2.0_f64))?,
                    builtins.getattr("complex")?.call1((f64::NAN, 0.0_f64))?,
                    builtins
                        .getattr("complex")?
                        .call1((1.0_f64, f64::INFINITY))?,
                ],
            )?,))?;
            let actual_complex = masked_invalid_fn.call1((complex_input.clone(),))?;
            let expected_complex = numpy_masked_invalid.call1((complex_input.clone(),))?;
            assert_eq!(repr_string(&actual_complex), repr_string(&expected_complex));

            let datetime_input = numpy.call_method(
                "array",
                (PyList::new(
                    py,
                    [
                        numpy.getattr("datetime64")?.call1(("NaT",))?,
                        numpy.getattr("datetime64")?.call1(("2020-01-01",))?,
                    ],
                )?,),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("dtype", "datetime64[D]")?;
                    kwargs
                }),
            )?;
            let actual_datetime = masked_invalid_fn.call1((datetime_input.clone(),))?;
            let expected_datetime = numpy_masked_invalid.call1((datetime_input.clone(),))?;
            assert_eq!(
                repr_string(&actual_datetime),
                repr_string(&expected_datetime)
            );

            let masked_input = numpy.getattr("ma")?.getattr("array")?.call(
                (vec![1.0_f64, f64::NAN],),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("mask", vec![false, true])?;
                    kwargs
                }),
            )?;
            let actual_masked = masked_invalid_fn.call1((masked_input.clone(),))?;
            let expected_masked = numpy_masked_invalid.call1((masked_input.clone(),))?;
            assert_eq!(repr_string(&actual_masked), repr_string(&expected_masked));

            let shared_input = numeric_array(py, vec![1.0_f64, f64::NAN], "float64");
            let actual_copy_false = masked_invalid_fn.call(
                (shared_input.clone(),),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("copy", false)?;
                    kwargs
                }),
            )?;
            let expected_copy_false = numpy_masked_invalid.call(
                (shared_input.clone(),),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("copy", false)?;
                    kwargs
                }),
            )?;
            let actual_shares = numpy
                .getattr("shares_memory")?
                .call1((actual_copy_false.getattr("data")?, shared_input.clone()))?
                .extract::<bool>()?;
            let expected_shares = numpy
                .getattr("shares_memory")?
                .call1((expected_copy_false.getattr("data")?, shared_input.clone()))?
                .extract::<bool>()?;
            assert_eq!(actual_shares, expected_shares);

            let object_values = PyList::new(
                py,
                [1_i64.into_pyobject(py)?.into_any().unbind(), py.None()],
            )?;
            let object_input = numpy.call_method(
                "array",
                (object_values,),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("dtype", numpy.getattr("object_")?)?;
                    kwargs
                }),
            )?;
            let actual_object_err = masked_invalid_fn
                .call1((object_input.clone(),))
                .unwrap_err();
            let expected_object_err = numpy_masked_invalid
                .call1((object_input.clone(),))
                .unwrap_err();
            assert_eq!(
                actual_object_err.get_type(py).name()?.extract::<String>()?,
                expected_object_err
                    .get_type(py)
                    .name()?
                    .extract::<String>()?
            );
            assert_eq!(
                actual_object_err.value(py).str()?.extract::<String>()?,
                expected_object_err.value(py).str()?.extract::<String>()?
            );

            Ok(())
        });
    }

    #[test]
    fn minimum_fill_value_matches_numpy_across_dtypes() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let minimum_fill_value_fn = module.getattr("minimum_fill_value")?;
            let numpy = py.import("numpy")?;
            let numpy_minimum_fill_value = numpy.getattr("ma")?.getattr("minimum_fill_value")?;

            // Array inputs across numeric, boolean, complex, and datetime dtypes.
            let float_arr = numeric_array(py, vec![1.0_f64, 2.0_f64], "float64");
            let actual_float = minimum_fill_value_fn.call1((float_arr.clone(),))?;
            let expected_float = numpy_minimum_fill_value.call1((float_arr.clone(),))?;
            assert_eq!(repr_string(&actual_float), repr_string(&expected_float));

            let int_arr = numeric_array(py, vec![1_i32, -2_i32], "int32");
            let actual_int = minimum_fill_value_fn.call1((int_arr.clone(),))?;
            let expected_int = numpy_minimum_fill_value.call1((int_arr.clone(),))?;
            assert_eq!(repr_string(&actual_int), repr_string(&expected_int));

            let uint_arr = numeric_array(py, vec![0_u64, 1_u64, 2_u64], "uint64");
            let actual_uint = minimum_fill_value_fn.call1((uint_arr.clone(),))?;
            let expected_uint = numpy_minimum_fill_value.call1((uint_arr.clone(),))?;
            assert_eq!(repr_string(&actual_uint), repr_string(&expected_uint));

            let bool_arr = numeric_array(py, vec![true, false], "bool");
            let actual_bool = minimum_fill_value_fn.call1((bool_arr.clone(),))?;
            let expected_bool = numpy_minimum_fill_value.call1((bool_arr.clone(),))?;
            assert_eq!(repr_string(&actual_bool), repr_string(&expected_bool));

            let builtins = py.import("builtins")?;
            let complex_arr = numpy.getattr("array")?.call(
                (PyList::new(
                    py,
                    [builtins.getattr("complex")?.call1((1.0_f64, 2.0_f64))?],
                )?,),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("dtype", "complex128")?;
                    kwargs
                }),
            )?;
            let actual_complex = minimum_fill_value_fn.call1((complex_arr.clone(),))?;
            let expected_complex = numpy_minimum_fill_value.call1((complex_arr.clone(),))?;
            assert_eq!(repr_string(&actual_complex), repr_string(&expected_complex));

            // Masked array input — dtype comes from the wrapped ndarray.
            let masked_input = numpy.getattr("ma")?.getattr("array")?.call(
                (vec![1.0_f64, 2.0_f64],),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("mask", vec![false, true])?;
                    kwargs
                }),
            )?;
            let actual_masked = minimum_fill_value_fn.call1((masked_input.clone(),))?;
            let expected_masked = numpy_minimum_fill_value.call1((masked_input.clone(),))?;
            assert_eq!(repr_string(&actual_masked), repr_string(&expected_masked));

            // Numpy dtype object directly (np.ma.minimum_fill_value also accepts a dtype).
            let dtype_obj = numpy.getattr("dtype")?.call1(("float32",))?;
            let actual_dtype = minimum_fill_value_fn.call1((dtype_obj.clone(),))?;
            let expected_dtype = numpy_minimum_fill_value.call1((dtype_obj.clone(),))?;
            assert_eq!(repr_string(&actual_dtype), repr_string(&expected_dtype));

            Ok(())
        });
    }

    #[test]
    fn minimum_fill_value_matches_numpy_supported_and_fallback_cases() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let minimum_fill_value_fn = module.getattr("minimum_fill_value")?;
            let numpy = py.import("numpy")?;
            let numpy_minimum_fill_value = numpy.getattr("ma")?.getattr("minimum_fill_value")?;

            let int_input = numeric_array(py, vec![1_i8, 2, 3], "int8");
            let actual_int = minimum_fill_value_fn.call1((int_input.clone(),))?;
            let expected_int = numpy_minimum_fill_value.call1((int_input.clone(),))?;
            assert_eq!(repr_string(&actual_int), repr_string(&expected_int));

            let float_input = numeric_array(py, vec![1.0_f64, 2.0], "float64");
            let actual_float = minimum_fill_value_fn.call1((float_input.clone(),))?;
            let expected_float = numpy_minimum_fill_value.call1((float_input.clone(),))?;
            assert_eq!(repr_string(&actual_float), repr_string(&expected_float));

            let builtins = py.import("builtins")?;
            let complex_input = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    builtins.getattr("complex")?.call1((1.0_f64, 2.0_f64))?,
                    builtins.getattr("complex")?.call1((3.0_f64, 4.0_f64))?,
                ],
            )?,))?;
            let actual_complex = minimum_fill_value_fn.call1((complex_input.clone(),))?;
            let expected_complex = numpy_minimum_fill_value.call1((complex_input.clone(),))?;
            assert_eq!(repr_string(&actual_complex), repr_string(&expected_complex));

            let datetime_input = numpy.call_method(
                "array",
                (PyList::new(
                    py,
                    [
                        numpy.getattr("datetime64")?.call1(("NaT",))?,
                        numpy.getattr("datetime64")?.call1(("2020-01-01",))?,
                    ],
                )?,),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("dtype", "datetime64[D]")?;
                    kwargs
                }),
            )?;
            let actual_datetime = minimum_fill_value_fn.call1((datetime_input.clone(),))?;
            let expected_datetime = numpy_minimum_fill_value.call1((datetime_input.clone(),))?;
            assert_eq!(
                repr_string(&actual_datetime),
                repr_string(&expected_datetime)
            );

            let object_input = numpy.call_method(
                "array",
                (PyList::new(py, [py.None()])?,),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("dtype", numpy.getattr("object_")?)?;
                    kwargs
                }),
            )?;
            let actual_object = minimum_fill_value_fn.call1((object_input.clone(),))?;
            let expected_object = numpy_minimum_fill_value.call1((object_input.clone(),))?;
            assert_eq!(repr_string(&actual_object), repr_string(&expected_object));

            let structured_input = numpy.call_method(
                "array",
                (PyList::new(py, [(1_i32, 2_i16)])?,),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("dtype", PyList::new(py, [("x", "i4"), ("y", "i2")])?)?;
                    kwargs
                }),
            )?;
            let actual_structured = minimum_fill_value_fn.call1((structured_input.clone(),))?;
            let expected_structured =
                numpy_minimum_fill_value.call1((structured_input.clone(),))?;
            assert_eq!(
                repr_string(&actual_structured),
                repr_string(&expected_structured)
            );

            let dtype_input = numpy.getattr("dtype")?.call1(("int32",))?;
            let actual_dtype = minimum_fill_value_fn.call1((dtype_input.clone(),))?;
            let expected_dtype = numpy_minimum_fill_value.call1((dtype_input,))?;
            assert_eq!(repr_string(&actual_dtype), repr_string(&expected_dtype));

            Ok(())
        });
    }

    #[test]
    fn maximum_fill_value_matches_numpy_across_dtypes() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let maximum_fill_value_fn = module.getattr("maximum_fill_value")?;
            let numpy = py.import("numpy")?;
            let numpy_maximum_fill_value = numpy.getattr("ma")?.getattr("maximum_fill_value")?;

            // Array inputs across numeric/boolean/complex/datetime dtypes.
            let float_arr = numeric_array(py, vec![1.0_f64, 2.0_f64], "float64");
            assert_eq!(
                repr_string(&maximum_fill_value_fn.call1((float_arr.clone(),))?),
                repr_string(&numpy_maximum_fill_value.call1((float_arr.clone(),))?)
            );

            let int_arr = numeric_array(py, vec![1_i32, -2_i32], "int32");
            assert_eq!(
                repr_string(&maximum_fill_value_fn.call1((int_arr.clone(),))?),
                repr_string(&numpy_maximum_fill_value.call1((int_arr.clone(),))?)
            );

            let uint_arr = numeric_array(py, vec![0_u64, 1_u64, 2_u64], "uint64");
            assert_eq!(
                repr_string(&maximum_fill_value_fn.call1((uint_arr.clone(),))?),
                repr_string(&numpy_maximum_fill_value.call1((uint_arr.clone(),))?)
            );

            let bool_arr = numeric_array(py, vec![true, false], "bool");
            assert_eq!(
                repr_string(&maximum_fill_value_fn.call1((bool_arr.clone(),))?),
                repr_string(&numpy_maximum_fill_value.call1((bool_arr.clone(),))?)
            );

            let builtins = py.import("builtins")?;
            let complex_arr = numpy.getattr("array")?.call(
                (PyList::new(
                    py,
                    [builtins.getattr("complex")?.call1((1.0_f64, 2.0_f64))?],
                )?,),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("dtype", "complex128")?;
                    kwargs
                }),
            )?;
            assert_eq!(
                repr_string(&maximum_fill_value_fn.call1((complex_arr.clone(),))?),
                repr_string(&numpy_maximum_fill_value.call1((complex_arr.clone(),))?)
            );

            // Masked array input — dtype comes from the wrapped ndarray.
            let masked_input = numpy.getattr("ma")?.getattr("array")?.call(
                (vec![1.0_f64, 2.0_f64],),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("mask", vec![false, true])?;
                    kwargs
                }),
            )?;
            assert_eq!(
                repr_string(&maximum_fill_value_fn.call1((masked_input.clone(),))?),
                repr_string(&numpy_maximum_fill_value.call1((masked_input.clone(),))?)
            );

            // Standalone np.dtype — ma.maximum_fill_value also accepts a dtype.
            let dtype_obj = numpy.getattr("dtype")?.call1(("float32",))?;
            assert_eq!(
                repr_string(&maximum_fill_value_fn.call1((dtype_obj.clone(),))?),
                repr_string(&numpy_maximum_fill_value.call1((dtype_obj.clone(),))?)
            );

            // datetime / timedelta must use -i64::MAX (one past NaT).
            let datetime_arr = numpy.call_method(
                "array",
                (PyList::new(
                    py,
                    [numpy.getattr("datetime64")?.call1(("2020-01-01",))?],
                )?,),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("dtype", "datetime64[D]")?;
                    kwargs
                }),
            )?;
            assert_eq!(
                repr_string(&maximum_fill_value_fn.call1((datetime_arr.clone(),))?),
                repr_string(&numpy_maximum_fill_value.call1((datetime_arr.clone(),))?)
            );

            Ok(())
        });
    }

    #[test]
    fn minimum_fill_value_supported_dtypes_do_not_delegate_to_numpy_ma() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let minimum_fill_value_fn = module.getattr("minimum_fill_value")?;
            let numpy = py.import("numpy")?;
            let ma = numpy.getattr("ma")?;
            let original = ma.getattr("minimum_fill_value")?;
            let bomb = py.eval(
                pyo3::ffi::c_str!(
                    "lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('should not delegate'))"
                ),
                None,
                None,
            )?;

            ma.setattr("minimum_fill_value", bomb)?;
            let result = (|| -> PyResult<()> {
                let int_input = numeric_array(py, vec![1_i8, 2, 3], "int8");
                let actual_int = minimum_fill_value_fn.call1((int_input,))?;
                assert_eq!(repr_string(&actual_int), "127");

                let dtype_input = numpy.getattr("dtype")?.call1(("int32",))?;
                let actual_dtype = minimum_fill_value_fn.call1((dtype_input,))?;
                assert_eq!(repr_string(&actual_dtype), "2147483647");
                Ok(())
            })();
            ma.setattr("minimum_fill_value", original)?;
            result
        });
    }

    #[test]
    fn pinv_matches_numpy_across_shapes_rcond_and_hermitian() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let pinv_fn = module.getattr("pinv")?;
            let numpy = py.import("numpy")?;
            let numpy_pinv = numpy.getattr("linalg")?.getattr("pinv")?;
            let allclose = numpy.getattr("allclose")?;

            // Square invertible 2x2.
            let square = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1.0_f64, 2.0])?,
                    PyList::new(py, [3.0_f64, 4.0])?,
                ],
            )?,))?;
            let actual = pinv_fn.call1((square.clone(),))?;
            let expected = numpy_pinv.call1((square.clone(),))?;
            assert!(
                allclose.call1((&actual, &expected))?.extract::<bool>()?,
                "square pinv diverged from numpy"
            );

            // Rectangular 3x2 (tall).
            let tall = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1.0_f64, 0.0])?,
                    PyList::new(py, [0.0_f64, 1.0])?,
                    PyList::new(py, [1.0_f64, 1.0])?,
                ],
            )?,))?;
            let actual_tall = pinv_fn.call1((tall.clone(),))?;
            let expected_tall = numpy_pinv.call1((tall.clone(),))?;
            assert!(
                allclose
                    .call1((&actual_tall, &expected_tall))?
                    .extract::<bool>()?,
                "tall pinv diverged from numpy"
            );

            // Rank-deficient square — pinv must stay finite and match numpy.
            let singular = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1.0_f64, 2.0])?,
                    PyList::new(py, [2.0_f64, 4.0])?,
                ],
            )?,))?;
            let actual_singular = pinv_fn.call1((singular.clone(),))?;
            let expected_singular = numpy_pinv.call1((singular.clone(),))?;
            assert!(
                allclose
                    .call1((&actual_singular, &expected_singular))?
                    .extract::<bool>()?,
                "singular pinv diverged from numpy"
            );

            // Explicit rcond override (scalar).
            let rcond_kwargs_1 = PyDict::new(py);
            rcond_kwargs_1.set_item("rcond", 1e-10_f64)?;
            let rcond_kwargs_2 = PyDict::new(py);
            rcond_kwargs_2.set_item("rcond", 1e-10_f64)?;
            let actual_rcond = pinv_fn.call((square.clone(),), Some(&rcond_kwargs_1))?;
            let expected_rcond = numpy_pinv.call((square.clone(),), Some(&rcond_kwargs_2))?;
            assert!(
                allclose
                    .call1((&actual_rcond, &expected_rcond))?
                    .extract::<bool>()?,
                "rcond-override pinv diverged from numpy"
            );

            // hermitian=True on a symmetric positive-definite 2x2.
            let spd = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [2.0_f64, 1.0])?,
                    PyList::new(py, [1.0_f64, 2.0])?,
                ],
            )?,))?;
            let herm_kwargs_1 = PyDict::new(py);
            herm_kwargs_1.set_item("hermitian", true)?;
            let herm_kwargs_2 = PyDict::new(py);
            herm_kwargs_2.set_item("hermitian", true)?;
            let actual_herm = pinv_fn.call((spd.clone(),), Some(&herm_kwargs_1))?;
            let expected_herm = numpy_pinv.call((spd.clone(),), Some(&herm_kwargs_2))?;
            assert!(
                allclose
                    .call1((&actual_herm, &expected_herm))?
                    .extract::<bool>()?,
                "hermitian pinv diverged from numpy"
            );

            Ok(())
        });
    }

    #[test]
    fn pinv_matches_numpy_rtol_keyword_only_and_error_surface() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let pinv_fn = module.getattr("pinv")?;
            let numpy = py.import("numpy")?;
            let numpy_pinv = numpy.getattr("linalg")?.getattr("pinv")?;
            let allclose = numpy.getattr("allclose")?;

            let assert_allclose = |actual: &pyo3::Bound<'_, PyAny>,
                                   expected: &pyo3::Bound<'_, PyAny>|
             -> PyResult<()> {
                let close = allclose.call1((actual, expected))?.extract::<bool>()?;
                assert!(close, "pinv result diverged from numpy");
                Ok(())
            };

            let matrix = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1.0_f64, 2.0])?,
                    PyList::new(py, [3.0_f64, 4.0])?,
                ],
            )?,))?;

            let actual_rtol = pinv_fn.call(
                (matrix.clone(),),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("rtol", 1e-12_f64)?;
                    kwargs
                }),
            )?;
            let expected_rtol = numpy_pinv.call(
                (matrix.clone(),),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("rtol", 1e-12_f64)?;
                    kwargs
                }),
            )?;
            assert_allclose(&actual_rtol, &expected_rtol)?;

            let actual_positional_err = pinv_fn
                .call1((matrix.clone(), py.None(), false, 1e-12_f64))
                .unwrap_err();
            let expected_positional_err = numpy_pinv
                .call1((matrix.clone(), py.None(), false, 1e-12_f64))
                .unwrap_err();
            assert_eq!(
                actual_positional_err
                    .get_type(py)
                    .name()?
                    .extract::<String>()?,
                expected_positional_err
                    .get_type(py)
                    .name()?
                    .extract::<String>()?
            );
            assert_eq!(
                actual_positional_err.value(py).str()?.extract::<String>()?,
                expected_positional_err
                    .value(py)
                    .str()?
                    .extract::<String>()?
            );

            let actual_conflict = call_outcome(
                py,
                &pinv_fn,
                &PyTuple::new(py, [matrix.clone()])?,
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("rcond", 1e-12_f64)?;
                    kwargs.set_item("rtol", py.None())?;
                    kwargs
                }),
            )?;
            let expected_conflict = call_outcome(
                py,
                &numpy_pinv,
                &PyTuple::new(py, [matrix.clone()])?,
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("rcond", 1e-12_f64)?;
                    kwargs.set_item("rtol", py.None())?;
                    kwargs
                }),
            )?;
            assert_eq!(actual_conflict, expected_conflict);

            Ok(())
        });
    }

    #[test]
    fn pinv_supported_real_inputs_do_not_delegate_to_numpy() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let pinv_fn = module.getattr("pinv")?;
            let numpy = py.import("numpy")?;
            let linalg = numpy.getattr("linalg")?;
            let allclose = numpy.getattr("allclose")?;
            let original = linalg.getattr("pinv")?;
            let bomb = py.eval(
                pyo3::ffi::c_str!(
                    "lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('should not delegate'))"
                ),
                None,
                None,
            )?;

            linalg.setattr("pinv", bomb)?;
            let result = (|| -> PyResult<()> {
                let matrix = numeric_array(py, vec![1.0_f64, 2.0, 3.0, 4.0], "float64")
                    .call_method1("reshape", ((2, 2),))?;
                let actual = pinv_fn.call1((matrix,))?;
                let expected = numeric_array(py, vec![-2.0_f64, 1.0, 1.5, -0.5], "float64")
                    .call_method1("reshape", ((2, 2),))?;
                if !allclose.call1((&actual, &expected))?.extract::<bool>()? {
                    return Err(PyValueError::new_err(
                        "pinv: local real 2-D path diverged from expected inverse",
                    ));
                }

                let hermitian_matrix = numeric_array(py, vec![2.0_f64, 1.0, 1.0, 2.0], "float64")
                    .call_method1("reshape", ((2, 2),))?;
                let actual_hermitian = pinv_fn.call(
                    (hermitian_matrix,),
                    Some(&{
                        let kwargs = PyDict::new(py);
                        kwargs.set_item("hermitian", true)?;
                        kwargs
                    }),
                )?;
                let expected_hermitian = numeric_array(
                    py,
                    vec![2.0_f64 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0],
                    "float64",
                )
                .call_method1("reshape", ((2, 2),))?;
                if !allclose
                    .call1((&actual_hermitian, &expected_hermitian))?
                    .extract::<bool>()?
                {
                    return Err(PyValueError::new_err(
                        "pinv: local hermitian path diverged from expected inverse",
                    ));
                }

                Ok(())
            })();
            linalg.setattr("pinv", original)?;
            result
        });
    }

    #[test]
    fn eigvals_matches_numpy_for_real_and_complex_spectra() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let eigvals_fn = module.getattr("eigvals")?;
            let numpy = py.import("numpy")?;
            let numpy_eigvals = numpy.getattr("linalg")?.getattr("eigvals")?;

            let diagonal = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [3.0_f64, 0.0])?,
                    PyList::new(py, [0.0_f64, 5.0])?,
                ],
            )?,))?;
            let actual_diagonal = eigvals_fn.call1((diagonal.clone(),))?;
            let expected_diagonal = numpy_eigvals.call1((diagonal.clone(),))?;
            assert_array_matches_numpy(&actual_diagonal, &expected_diagonal)?;

            let rotation = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [0.0_f64, -1.0])?,
                    PyList::new(py, [1.0_f64, 0.0])?,
                ],
            )?,))?;
            let actual_rotation = eigvals_fn.call1((rotation.clone(),))?;
            let expected_rotation = numpy_eigvals.call1((rotation,))?;
            assert_array_matches_numpy(&actual_rotation, &expected_rotation)?;

            Ok(())
        });
    }

    #[test]
    fn eigvals_supported_real_square_inputs_do_not_delegate_to_numpy() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let eigvals_fn = module.getattr("eigvals")?;
            let numpy = py.import("numpy")?;
            let linalg = numpy.getattr("linalg")?;
            let builtins = py.import("builtins")?;
            let original = linalg.getattr("eigvals")?;
            let bomb = py.eval(
                pyo3::ffi::c_str!(
                    "lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('should not delegate'))"
                ),
                None,
                None,
            )?;

            linalg.setattr("eigvals", bomb)?;
            let result = (|| -> PyResult<()> {
                let matrix = numeric_array(py, vec![0.0_f64, -1.0, 1.0, 0.0], "float64")
                    .call_method1("reshape", ((2, 2),))?;
                let actual = eigvals_fn.call1((matrix,))?;
                let expected = numpy.getattr("array")?.call1((PyList::new(
                    py,
                    [
                        builtins.getattr("complex")?.call1((0.0_f64, 1.0))?,
                        builtins.getattr("complex")?.call1((0.0_f64, -1.0))?,
                    ],
                )?,))?;
                assert_array_matches_numpy(&actual, &expected)?;
                Ok(())
            })();
            linalg.setattr("eigvals", original)?;
            result
        });
    }

    #[test]
    fn expand_dims_matches_numpy_list_axes_and_repeated_axis_error() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let expand_dims_fn = module.getattr("expand_dims")?;
            let numpy_expand_dims = py.import("numpy")?.getattr("expand_dims")?;

            let values = numeric_array(py, vec![1_i64, 2, 3], "int64");
            let list_axes = PyList::new(py, [0_isize, 2_isize])?;
            let actual_list = expand_dims_fn.call1((values.clone(), list_axes))?;
            let expected_list =
                numpy_expand_dims.call1((values.clone(), PyList::new(py, [0_isize, 2_isize])?))?;
            assert_array_matches_numpy(&actual_list, &expected_list)?;

            let negative_axes = PyList::new(py, [-1_isize, -2_isize])?;
            let actual_negative = expand_dims_fn.call1((values.clone(), negative_axes))?;
            let expected_negative = numpy_expand_dims
                .call1((values.clone(), PyList::new(py, [-1_isize, -2_isize])?))?;
            assert_array_matches_numpy(&actual_negative, &expected_negative)?;

            let actual_empty = expand_dims_fn.call1((values.clone(), PyList::empty(py)))?;
            let expected_empty = numpy_expand_dims.call1((values.clone(), PyList::empty(py)))?;
            assert_array_matches_numpy(&actual_empty, &expected_empty)?;

            let repeated_axis = PyList::new(py, [0_isize, 0_isize])?;
            let actual_err = expand_dims_fn
                .call1((values.clone(), repeated_axis))
                .unwrap_err();
            let expected_err = numpy_expand_dims
                .call1((values, PyList::new(py, [0_isize, 0_isize])?))
                .unwrap_err();
            assert_eq!(
                actual_err.get_type(py).name()?.extract::<String>()?,
                expected_err.get_type(py).name()?.extract::<String>()?
            );
            assert_eq!(
                actual_err.value(py).str()?.extract::<String>()?,
                expected_err.value(py).str()?.extract::<String>()?
            );

            Ok(())
        });
    }

    #[test]
    fn structured_to_unstructured_matches_numpy_mixed_dtype_and_dtype_kwarg() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let numpy = py.import("numpy")?;
            let recfunctions = py.import("numpy.lib.recfunctions")?;

            let mixed_dtype = numpy
                .getattr("dtype")?
                .call1((PyList::new(py, [("x", "<i4"), ("y", "<f8")])?,))?;
            let array_kwargs = PyDict::new(py);
            array_kwargs.set_item("dtype", mixed_dtype.clone())?;
            let structured = numpy.getattr("array")?.call(
                (PyList::new(py, [(1_i32, 2.5_f64), (3_i32, 4.5_f64)])?,),
                Some(&array_kwargs),
            )?;

            let actual_default = module
                .getattr("structured_to_unstructured")?
                .call1((structured.clone(),))?;
            let expected_default = recfunctions
                .getattr("structured_to_unstructured")?
                .call1((structured.clone(),))?;
            assert_array_matches_numpy(&actual_default, &expected_default)?;

            let typed_kwargs = PyDict::new(py);
            typed_kwargs.set_item("dtype", numpy.getattr("float32")?)?;
            typed_kwargs.set_item("casting", "unsafe")?;
            let actual_typed = module
                .getattr("structured_to_unstructured")?
                .call((structured.clone(),), Some(&typed_kwargs))?;
            let expected_typed = recfunctions
                .getattr("structured_to_unstructured")?
                .call((structured.clone(),), Some(&typed_kwargs))?;
            assert_array_matches_numpy(&actual_typed, &expected_typed)?;

            let scalar = numpy
                .getattr("array")?
                .call((PyList::new(py, [(1_i32, 2.5_f64)])?,), Some(&array_kwargs))?
                .get_item(0)?;
            let actual_scalar = module
                .getattr("structured_to_unstructured")?
                .call1((scalar.clone(),))?;
            let expected_scalar = recfunctions
                .getattr("structured_to_unstructured")?
                .call1((scalar.clone(),))?;
            assert_array_matches_numpy(&actual_scalar, &expected_scalar)?;

            Ok(())
        });
    }

    #[test]
    fn structured_to_unstructured_matches_numpy_subarray_fields() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let numpy = py.import("numpy")?;
            let recfunctions = py.import("numpy.lib.recfunctions")?;

            let subarray_dtype = numpy.getattr("dtype")?.call1((py.eval(
                pyo3::ffi::c_str!("[('xy', '<i4', (2,)), ('z', '<i4')]"),
                None,
                None,
            )?,))?;
            let array_kwargs = PyDict::new(py);
            array_kwargs.set_item("dtype", subarray_dtype)?;
            let structured = numpy.getattr("array")?.call(
                (PyList::new(
                    py,
                    [(vec![1_i32, 2_i32], 3_i32), (vec![4_i32, 5_i32], 6_i32)],
                )?,),
                Some(&array_kwargs),
            )?;

            let actual = module
                .getattr("structured_to_unstructured")?
                .call1((structured.clone(),))?;
            let expected = recfunctions
                .getattr("structured_to_unstructured")?
                .call1((structured.clone(),))?;
            assert_array_matches_numpy(&actual, &expected)?;

            Ok(())
        });
    }

    #[test]
    fn solve_triangular_matches_numpy_unit_diagonal_variants() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let numpy = py.import("numpy")?;
            let linalg = numpy.getattr("linalg")?;
            let eye3 = numpy.getattr("eye")?.call1((3,))?;

            let lower = numeric_array(
                py,
                vec![
                    vec![9.0, 0.0, 0.0],
                    vec![2.0, -7.0, 0.0],
                    vec![4.0, -3.0, 11.0],
                ],
                "float64",
            );
            let lower_rhs = numeric_array(py, vec![1.0, 2.0, 3.0], "float64");
            let lower_actual = solve_triangular(
                py,
                lower.clone().unbind(),
                lower_rhs.clone().unbind(),
                true,
                true,
            )?;
            let lower_unit = numpy
                .call_method1("tril", (lower.clone(), -1_i32))?
                .call_method1("__add__", (eye3.clone(),))?;
            let lower_expected = linalg.call_method1("solve", (lower_unit, lower_rhs))?;
            assert_array_matches_numpy(lower_actual.bind(py), &lower_expected)?;

            let upper = numeric_array(
                py,
                vec![
                    vec![5.0, 2.0, -1.0],
                    vec![0.0, 8.0, 3.0],
                    vec![0.0, 0.0, -4.0],
                ],
                "float64",
            );
            let upper_rhs = numeric_array(py, vec![4.0, 5.0, 6.0], "float64");
            let upper_actual = solve_triangular(
                py,
                upper.clone().unbind(),
                upper_rhs.clone().unbind(),
                false,
                true,
            )?;
            let upper_unit = numpy
                .call_method1("triu", (upper.clone(), 1_i32))?
                .call_method1("__add__", (eye3,))?;
            let upper_expected = linalg.call_method1("solve", (upper_unit, upper_rhs.clone()))?;
            assert_array_matches_numpy(upper_actual.bind(py), &upper_expected)?;

            let upper_full_actual = solve_triangular(
                py,
                upper.clone().unbind(),
                upper_rhs.clone().unbind(),
                false,
                false,
            )?;
            let upper_full_expected = linalg.call_method1("solve", (upper, upper_rhs))?;
            assert_array_matches_numpy(upper_full_actual.bind(py), &upper_full_expected)
        });
    }

    #[test]
    fn count_nonzero_matches_numpy_empty_array_keepdims_variants() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let numpy = py.import("numpy")?;

            let empty_rows = numeric_array(py, Vec::<i64>::new(), "int64")
                .call_method1("reshape", (0_usize, 3_usize))?;
            let axis0 = 0_i32.into_pyobject(py)?.unbind();
            let actual_axis0 =
                count_nonzero(py, empty_rows.clone().unbind(), Some(axis0.into()), true)?;
            let expected_axis0 = numpy.call_method(
                "count_nonzero",
                (empty_rows.clone(),),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("axis", 0)?;
                    kwargs.set_item("keepdims", true)?;
                    kwargs
                }),
            )?;

            assert_eq!(
                actual_axis0
                    .bind(py)
                    .getattr("shape")?
                    .extract::<Vec<usize>>()?,
                expected_axis0.getattr("shape")?.extract::<Vec<usize>>()?
            );
            assert_eq!(
                repr_string(&actual_axis0.bind(py).call_method0("tolist")?),
                repr_string(&expected_axis0.call_method0("tolist")?)
            );

            let empty_cols = numeric_array(py, vec![Vec::<i64>::new(), Vec::<i64>::new()], "int64");
            let axis1 = 1_i32.into_pyobject(py)?.unbind();
            let actual_axis1 =
                count_nonzero(py, empty_cols.clone().unbind(), Some(axis1.into()), true)?;
            let expected_axis1 = numpy.call_method(
                "count_nonzero",
                (empty_cols.clone(),),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("axis", 1)?;
                    kwargs.set_item("keepdims", true)?;
                    kwargs
                }),
            )?;

            assert_eq!(
                actual_axis1
                    .bind(py)
                    .getattr("shape")?
                    .extract::<Vec<usize>>()?,
                expected_axis1.getattr("shape")?.extract::<Vec<usize>>()?
            );
            assert_eq!(
                repr_string(&actual_axis1.bind(py).call_method0("tolist")?),
                repr_string(&expected_axis1.call_method0("tolist")?)
            );

            let axis_tuple = PyTuple::new(py, [0_isize, 1_isize])?.unbind();
            let actual_tuple = count_nonzero(
                py,
                empty_rows.clone().unbind(),
                Some(axis_tuple.clone_ref(py).into()),
                true,
            )?;
            let expected_tuple = numpy.call_method(
                "count_nonzero",
                (empty_rows,),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("axis", axis_tuple.bind(py))?;
                    kwargs.set_item("keepdims", true)?;
                    kwargs
                }),
            )?;

            assert_eq!(
                actual_tuple
                    .bind(py)
                    .getattr("shape")?
                    .extract::<Vec<usize>>()?,
                expected_tuple.getattr("shape")?.extract::<Vec<usize>>()?
            );
            assert_eq!(
                repr_string(&actual_tuple.bind(py).call_method0("tolist")?),
                repr_string(&expected_tuple.call_method0("tolist")?)
            );

            Ok(())
        });
    }

    #[test]
    fn count_nonzero_matches_numpy_empty_array_keepdims_extra_variants() {
        // Extends the primary empty+keepdims test with broader parity coverage:
        // 1-D empty input, axis=None on 2-D empty, non-keepdims on 2-D empty,
        // and 3-D empty arrays where the zero-size dim is interior. Each case
        // locks both the output shape and the flattened payload against
        // NumPy's oracle output.
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let numpy = py.import("numpy")?;

            let assert_matches = |py: Python<'_>,
                                  input: &pyo3::Bound<'_, pyo3::types::PyAny>,
                                  axis: Option<Py<PyAny>>,
                                  keepdims: bool,
                                  label: &str|
             -> PyResult<()> {
                let actual = count_nonzero(
                    py,
                    input.clone().unbind(),
                    axis.as_ref().map(|a| a.clone_ref(py)),
                    keepdims,
                )?;
                let kwargs = PyDict::new(py);
                if let Some(ref ax) = axis {
                    kwargs.set_item("axis", ax.bind(py))?;
                } else {
                    kwargs.set_item("axis", py.None())?;
                }
                kwargs.set_item("keepdims", keepdims)?;
                let expected =
                    numpy.call_method("count_nonzero", (input.clone(),), Some(&kwargs))?;

                let actual_bound = actual.bind(py);
                let actual_shape = actual_bound.getattr("shape")?.extract::<Vec<usize>>()?;
                let expected_shape = expected.getattr("shape")?.extract::<Vec<usize>>()?;
                assert_eq!(actual_shape, expected_shape, "{label}: shape mismatch");
                assert_eq!(
                    repr_string(&actual_bound.call_method0("tolist")?),
                    repr_string(&expected.call_method0("tolist")?),
                    "{label}: values mismatch"
                );
                Ok(())
            };

            let empty_1d = numeric_array(py, Vec::<i64>::new(), "int64");
            let axis0 = 0_i32.into_pyobject(py)?.unbind();
            assert_matches(
                py,
                &empty_1d,
                Some(axis0.into()),
                true,
                "1d_empty_axis0_keepdims",
            )?;
            assert_matches(py, &empty_1d, None, true, "1d_empty_axis_none_keepdims")?;

            let empty_rows = numeric_array(py, Vec::<i64>::new(), "int64")
                .call_method1("reshape", (0_usize, 3_usize))?;
            assert_matches(
                py,
                &empty_rows,
                None,
                true,
                "2d_empty_rows_axis_none_keepdims",
            )?;
            let axis0_2d = 0_i32.into_pyobject(py)?.unbind();
            assert_matches(
                py,
                &empty_rows,
                Some(axis0_2d.into()),
                false,
                "2d_empty_rows_axis0_no_keepdims",
            )?;
            let axis1_2d = 1_i32.into_pyobject(py)?.unbind();
            assert_matches(
                py,
                &empty_rows,
                Some(axis1_2d.into()),
                true,
                "2d_empty_rows_axis1_keepdims",
            )?;

            let empty_3d = numeric_array(py, Vec::<i64>::new(), "int64")
                .call_method1("reshape", (2_usize, 0_usize, 3_usize))?;
            let axis0_3d = 0_i32.into_pyobject(py)?.unbind();
            assert_matches(
                py,
                &empty_3d,
                Some(axis0_3d.into()),
                true,
                "3d_empty_middle_axis0_keepdims",
            )?;
            let axis_tuple = PyTuple::new(py, [0_isize, 2_isize])?.unbind();
            assert_matches(
                py,
                &empty_3d,
                Some(axis_tuple.into()),
                true,
                "3d_empty_middle_axis_tuple_keepdims",
            )?;
            assert_matches(
                py,
                &empty_3d,
                None,
                true,
                "3d_empty_middle_axis_none_keepdims",
            )?;

            Ok(())
        });
    }

    #[test]
    fn isposinf_matches_numpy() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let values = numeric_array(
                py,
                vec![f64::NEG_INFINITY, -1.0, 0.0, f64::INFINITY, f64::NAN],
                "float64",
            );
            let actual = isposinf(py, values.clone().unbind())?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("isposinf", (values,))?;

            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn isneginf_matches_numpy_multidimensional_input() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let values = numeric_array(
                py,
                vec![
                    vec![f64::NEG_INFINITY, 1.0, f64::INFINITY],
                    vec![0.0, f64::NEG_INFINITY, f64::NAN],
                ],
                "float64",
            );
            let actual = isneginf(py, values.clone().unbind())?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("isneginf", (values,))?;

            assert_eq!(
                actual.bind(py).getattr("shape")?.extract::<Vec<usize>>()?,
                expected.getattr("shape")?.extract::<Vec<usize>>()?
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn signbit_matches_numpy_with_signed_zero_and_nan() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let values = numeric_array(
                py,
                vec![
                    -0.0,
                    0.0,
                    -1.0,
                    2.5,
                    f64::NEG_INFINITY,
                    f64::INFINITY,
                    f64::NAN,
                ],
                "float64",
            );
            let actual = signbit(py, values.clone().unbind())?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("signbit", (values,))?;

            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn signbit_matches_numpy_multidimensional_input() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let values = numeric_array(
                py,
                vec![vec![-0.0, 1.0, -2.0], vec![3.0, -4.0, f64::NEG_INFINITY]],
                "float64",
            );
            let actual = signbit(py, values.clone().unbind())?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("signbit", (values,))?;

            assert_eq!(
                actual.bind(py).getattr("shape")?.extract::<Vec<usize>>()?,
                expected.getattr("shape")?.extract::<Vec<usize>>()?
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn isnan_matches_numpy() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let values = numeric_array(
                py,
                vec![f64::NAN, f64::NEG_INFINITY, -1.0, 0.0, f64::INFINITY],
                "float64",
            );
            let actual = isnan(py, values.clone().unbind())?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("isnan", (values,))?;

            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn isinf_matches_numpy_multidimensional_input() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let values = numeric_array(
                py,
                vec![
                    vec![f64::NEG_INFINITY, -1.0, 0.0],
                    vec![1.0, f64::INFINITY, f64::NAN],
                ],
                "float64",
            );
            let actual = isinf(py, values.clone().unbind())?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("isinf", (values,))?;

            assert_eq!(
                actual.bind(py).getattr("shape")?.extract::<Vec<usize>>()?,
                expected.getattr("shape")?.extract::<Vec<usize>>()?
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn isfinite_matches_numpy() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let values = numeric_array(
                py,
                vec![f64::NAN, f64::NEG_INFINITY, -3.0, -0.0, 1.5, f64::INFINITY],
                "float64",
            );
            let actual = isfinite(py, values.clone().unbind())?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("isfinite", (values,))?;

            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn spacing_matches_numpy_basic() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let values = numeric_array(py, vec![1.0, 0.0, -1.0, 2.0], "float64");
            let actual = spacing(py, values.clone().unbind())?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("spacing", (values,))?;

            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn spacing_matches_numpy_nan_and_inf_behavior() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let values = numeric_array(
                py,
                vec![f64::NAN, f64::NEG_INFINITY, f64::INFINITY, -0.0],
                "float64",
            );
            let actual = spacing(py, values.clone().unbind())?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("spacing", (values,))?;

            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn sign_matches_numpy_signed_zero_and_nan() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let values = numeric_array(
                py,
                vec![
                    -0.0,
                    0.0,
                    -2.5,
                    3.0,
                    f64::NAN,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                ],
                "float64",
            );
            let actual = sign(py, values.clone().unbind())?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("sign", (values,))?;

            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn sign_matches_numpy_multidimensional_integer_input() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let values = numeric_array(
                py,
                vec![vec![-2_i64, 0_i64, 5_i64], vec![7_i64, -11_i64, 0_i64]],
                "int64",
            );
            let actual = sign(py, values.clone().unbind())?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("sign", (values,))?;

            assert_eq!(
                actual.bind(py).getattr("shape")?.extract::<Vec<usize>>()?,
                expected.getattr("shape")?.extract::<Vec<usize>>()?
            );
            assert_eq!(
                actual
                    .bind(py)
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?,
                expected.getattr("dtype")?.str()?.extract::<String>()?
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn floor_matches_numpy_float_and_special_values() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let values = numeric_array(
                py,
                vec![-1.7, -0.2, -0.0, 0.0, 1.2, f64::INFINITY, f64::NAN],
                "float64",
            );
            let actual = floor(py, values.clone().unbind())?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("floor", (values,))?;

            assert_eq!(
                actual
                    .bind(py)
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?,
                expected.getattr("dtype")?.str()?.extract::<String>()?
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn ceil_matches_numpy_signed_zero_and_integer_input() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let float_values = numeric_array(
                py,
                vec![-1.7, -0.2, -0.0, 0.0, 1.2, f64::INFINITY, f64::NAN],
                "float64",
            );
            let actual_float = ceil(py, float_values.clone().unbind())?;
            let numpy = py.import("numpy")?;
            let expected_float = numpy.call_method1("ceil", (float_values,))?;

            assert_eq!(
                repr_string(&actual_float.bind(py).call_method0("tolist")?),
                repr_string(&expected_float.call_method0("tolist")?)
            );

            let int_values = numeric_array(
                py,
                vec![vec![1_i64, -2_i64, 0_i64], vec![5_i64, -8_i64, 3_i64]],
                "int64",
            );
            let actual_int = ceil(py, int_values.clone().unbind())?;
            let expected_int = numpy.call_method1("ceil", (int_values,))?;

            assert_eq!(
                actual_int
                    .bind(py)
                    .getattr("shape")?
                    .extract::<Vec<usize>>()?,
                expected_int.getattr("shape")?.extract::<Vec<usize>>()?
            );
            assert_eq!(
                actual_int
                    .bind(py)
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?,
                expected_int.getattr("dtype")?.str()?.extract::<String>()?
            );
            assert_eq!(
                repr_string(&actual_int.bind(py).call_method0("tolist")?),
                repr_string(&expected_int.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn degrees_matches_numpy_basic() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let values = numeric_array(
                py,
                vec![0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI],
                "float64",
            );
            let actual = degrees(py, values.clone().unbind())?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("degrees", (values,))?;

            assert_eq!(
                actual
                    .bind(py)
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?,
                expected.getattr("dtype")?.str()?.extract::<String>()?
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn trunc_matches_numpy_float_and_special_values() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let values = numeric_array(
                py,
                vec![
                    -0.0_f64,
                    0.0_f64,
                    1.7_f64,
                    -1.7_f64,
                    f64::INFINITY,
                    f64::NAN,
                ],
                "float64",
            );
            let actual = trunc(py, values.clone().unbind())?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("trunc", (values,))?;

            assert_eq!(
                actual
                    .bind(py)
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?,
                expected.getattr("dtype")?.str()?.extract::<String>()?
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn trunc_matches_numpy_integer_input() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let values = numeric_array(
                py,
                vec![vec![1_i64, -2_i64, 0_i64], vec![5_i64, -8_i64, 3_i64]],
                "int64",
            );
            let actual = trunc(py, values.clone().unbind())?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("trunc", (values,))?;

            assert_eq!(
                actual.bind(py).getattr("shape")?.extract::<Vec<usize>>()?,
                expected.getattr("shape")?.extract::<Vec<usize>>()?
            );
            assert_eq!(
                actual
                    .bind(py)
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?,
                expected.getattr("dtype")?.str()?.extract::<String>()?
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn rint_matches_numpy_float_and_special_values() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let values = numeric_array(
                py,
                vec![
                    -0.5_f64,
                    -0.0_f64,
                    0.0_f64,
                    0.5_f64,
                    1.5_f64,
                    -1.5_f64,
                    f64::INFINITY,
                    f64::NAN,
                ],
                "float64",
            );
            let actual = rint(py, values.clone().unbind())?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("rint", (values,))?;

            assert_eq!(
                actual
                    .bind(py)
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?,
                expected.getattr("dtype")?.str()?.extract::<String>()?
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn rint_matches_numpy_integer_input() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let values = numeric_array(
                py,
                vec![vec![1_i64, -2_i64, 0_i64], vec![7_i64, -9_i64, 4_i64]],
                "int64",
            );
            let actual = rint(py, values.clone().unbind())?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("rint", (values,))?;

            assert_eq!(
                actual.bind(py).getattr("shape")?.extract::<Vec<usize>>()?,
                expected.getattr("shape")?.extract::<Vec<usize>>()?
            );
            assert_eq!(
                actual
                    .bind(py)
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?,
                expected.getattr("dtype")?.str()?.extract::<String>()?
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn rint_matches_numpy_bool_input() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let values = numeric_array(py, vec![true, false, true], "bool");
            let actual = rint(py, values.clone().unbind())?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("rint", (values,))?;

            assert_eq!(
                actual
                    .bind(py)
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?,
                expected.getattr("dtype")?.str()?.extract::<String>()?
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn radians_matches_numpy_integer_multidimensional_input() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let values = numeric_array(
                py,
                vec![vec![0_i64, 90_i64], vec![180_i64, -45_i64]],
                "int64",
            );
            let actual = radians(py, values.clone().unbind())?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("radians", (values,))?;

            assert_eq!(
                actual.bind(py).getattr("shape")?.extract::<Vec<usize>>()?,
                expected.getattr("shape")?.extract::<Vec<usize>>()?
            );
            assert_eq!(
                actual
                    .bind(py)
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?,
                expected.getattr("dtype")?.str()?.extract::<String>()?
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn sinc_matches_numpy_basic() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let values = numeric_array(py, vec![0.0, 0.5, 1.0, -1.5], "float64");
            let actual = sinc(py, values.clone().unbind())?;

            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("sinc")?.call1((values,))?;

            assert_eq!(
                actual
                    .bind(py)
                    .call_method0("tolist")?
                    .extract::<Vec<f64>>()?,
                expected.call_method0("tolist")?.extract::<Vec<f64>>()?
            );
            Ok(())
        });
    }

    #[test]
    fn sinc_matches_numpy_multidimensional_integer_input() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let values = numeric_array(py, vec![vec![0_i64, 1], vec![-2, 3]], "int64");
            let actual = sinc(py, values.clone().unbind())?;

            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("sinc")?.call1((values,))?;

            assert_eq!(
                actual.bind(py).getattr("shape")?.extract::<Vec<usize>>()?,
                vec![2, 2]
            );
            assert_eq!(
                actual
                    .bind(py)
                    .call_method0("tolist")?
                    .extract::<Vec<Vec<f64>>>()?,
                expected
                    .call_method0("tolist")?
                    .extract::<Vec<Vec<f64>>>()?
            );
            Ok(())
        });
    }

    #[test]
    fn copysign_matches_numpy_basic() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let x1 = numeric_array(py, vec![1.0, -2.5, 3.0], "float64");
            let x2 = numeric_array(py, vec![-4.0, 5.0, -6.0], "float64");
            let actual = copysign(py, x1.clone().unbind(), x2.clone().unbind())?;

            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("copysign")?.call1((x1, x2))?;

            assert_eq!(
                actual
                    .bind(py)
                    .call_method0("tolist")?
                    .extract::<Vec<f64>>()?,
                expected.call_method0("tolist")?.extract::<Vec<f64>>()?
            );
            Ok(())
        });
    }

    #[test]
    fn copysign_matches_numpy_signed_zero_and_broadcasting() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let x1 = numeric_array(py, vec![vec![1.0, -1.0], vec![0.0, -0.0]], "float64");
            let x2 = numeric_array(py, vec![-0.0, 0.0], "float64");
            let actual = copysign(py, x1.clone().unbind(), x2.clone().unbind())?;

            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("copysign")?.call1((x1, x2))?;

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
    fn nextafter_matches_numpy_basic() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let x1 = numeric_array(py, vec![1.0, 0.0, -1.0], "float64");
            let x2 = numeric_array(py, vec![2.0, -1.0, 0.0], "float64");
            let actual = nextafter(py, x1.clone().unbind(), x2.clone().unbind())?;

            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("nextafter")?.call1((x1, x2))?;

            assert_eq!(
                actual
                    .bind(py)
                    .call_method0("tolist")?
                    .extract::<Vec<f64>>()?,
                expected.call_method0("tolist")?.extract::<Vec<f64>>()?
            );
            Ok(())
        });
    }

    #[test]
    fn nextafter_matches_numpy_signed_zero_and_broadcasting() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let x1 = numeric_array(py, vec![vec![0.0, -0.0], vec![1.0, -1.0]], "float64");
            let x2 = numeric_array(py, vec![-0.0, 0.0], "float64");
            let actual = nextafter(py, x1.clone().unbind(), x2.clone().unbind())?;

            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("nextafter")?.call1((x1, x2))?;

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
    fn hypot_matches_numpy_basic() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let x1 = numeric_array(py, vec![3.0, 5.0, 8.0], "float64");
            let x2 = numeric_array(py, vec![4.0, 12.0, 15.0], "float64");
            let actual = hypot(py, x1.clone().unbind(), x2.clone().unbind())?;

            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("hypot")?.call1((x1, x2))?;

            assert_eq!(
                actual
                    .bind(py)
                    .call_method0("tolist")?
                    .extract::<Vec<f64>>()?,
                expected.call_method0("tolist")?.extract::<Vec<f64>>()?
            );
            Ok(())
        });
    }

    #[test]
    fn hypot_matches_numpy_broadcasting() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let x1 = numeric_array(py, vec![vec![3.0, 6.0], vec![8.0, 0.0]], "float64");
            let x2 = numeric_array(py, vec![4.0, 8.0], "float64");
            let actual = hypot(py, x1.clone().unbind(), x2.clone().unbind())?;

            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("hypot")?.call1((x1, x2))?;

            assert_eq!(
                actual.bind(py).getattr("shape")?.extract::<Vec<usize>>()?,
                vec![2, 2]
            );
            assert_eq!(
                actual
                    .bind(py)
                    .call_method0("tolist")?
                    .extract::<Vec<Vec<f64>>>()?,
                expected
                    .call_method0("tolist")?
                    .extract::<Vec<Vec<f64>>>()?
            );
            Ok(())
        });
    }

    #[test]
    fn ldexp_matches_numpy_basic() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let x1 = numeric_array(py, vec![0.5, 1.5, -2.0], "float64");
            let x2 = numeric_array(py, vec![1_i64, 2_i64, 3_i64], "int64");
            let actual = ldexp(py, x1.clone().unbind(), x2.clone().unbind())?;

            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("ldexp")?.call1((x1, x2))?;

            assert_eq!(
                actual
                    .bind(py)
                    .call_method0("tolist")?
                    .extract::<Vec<f64>>()?,
                expected.call_method0("tolist")?.extract::<Vec<f64>>()?
            );
            Ok(())
        });
    }

    #[test]
    fn ldexp_matches_numpy_broadcasting_and_zero_sign() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let x1 = numeric_array(py, vec![vec![0.0, -0.0], vec![0.5, -1.5]], "float64");
            let x2 = numeric_array(py, vec![1_i64, 2_i64], "int64");
            let actual = ldexp(py, x1.clone().unbind(), x2.clone().unbind())?;

            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("ldexp")?.call1((x1, x2))?;

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
    fn logaddexp_matches_numpy_basic() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let x1 = numeric_array(py, vec![0.0, 1.0, -2.0], "float64");
            let x2 = numeric_array(py, vec![0.0, 3.0, -4.0], "float64");
            let actual = logaddexp(py, x1.clone().unbind(), x2.clone().unbind())?;

            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("logaddexp")?.call1((x1, x2))?;
            let actual_values = actual
                .bind(py)
                .call_method0("tolist")?
                .extract::<Vec<f64>>()?;
            let expected_values = expected.call_method0("tolist")?.extract::<Vec<f64>>()?;

            assert_eq!(actual_values.len(), expected_values.len());
            for (actual, expected) in actual_values.iter().zip(expected_values.iter()) {
                assert!(
                    (actual - expected).abs() <= 1e-15,
                    "expected {expected}, got {actual}"
                );
            }
            Ok(())
        });
    }

    #[test]
    fn logaddexp_matches_numpy_broadcasting_and_infinities() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let x1 = numeric_array(
                py,
                vec![vec![f64::NEG_INFINITY, 1.0], vec![2.0, 3.0]],
                "float64",
            );
            let x2 = numeric_array(py, vec![f64::NEG_INFINITY, f64::INFINITY], "float64");
            let actual = logaddexp(py, x1.clone().unbind(), x2.clone().unbind())?;

            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("logaddexp")?.call1((x1, x2))?;

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
    fn logaddexp2_matches_numpy_basic() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let x1 = numeric_array(py, vec![0.0, 1.0, -2.0], "float64");
            let x2 = numeric_array(py, vec![0.0, 3.0, -4.0], "float64");
            let actual = logaddexp2(py, x1.clone().unbind(), x2.clone().unbind())?;

            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("logaddexp2")?.call1((x1, x2))?;
            let actual_values = actual
                .bind(py)
                .call_method0("tolist")?
                .extract::<Vec<f64>>()?;
            let expected_values = expected.call_method0("tolist")?.extract::<Vec<f64>>()?;

            assert_eq!(actual_values.len(), expected_values.len());
            for (actual, expected) in actual_values.iter().zip(expected_values.iter()) {
                assert!(
                    (actual - expected).abs() <= 1e-15,
                    "expected {expected}, got {actual}"
                );
            }
            Ok(())
        });
    }

    #[test]
    fn logaddexp2_matches_numpy_broadcasting_and_infinities() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let x1 = numeric_array(
                py,
                vec![vec![f64::NEG_INFINITY, 1.0], vec![2.0, 3.0]],
                "float64",
            );
            let x2 = numeric_array(py, vec![f64::NEG_INFINITY, f64::INFINITY], "float64");
            let actual = logaddexp2(py, x1.clone().unbind(), x2.clone().unbind())?;

            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("logaddexp2")?.call1((x1, x2))?;

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
    fn frexp_matches_numpy_basic() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let x = numeric_array(py, vec![0.0, 1.0, -2.5, 8.0], "float64");
            let actual = frexp(py, x.clone().unbind())?;

            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("frexp")?.call1((x,))?;

            let actual_tuple = actual.bind(py).downcast::<PyTuple>()?;
            let expected_tuple = expected.downcast::<PyTuple>()?;
            assert_eq!(actual_tuple.len()?, 2);
            assert_eq!(expected_tuple.len()?, 2);

            let actual_mantissa = actual_tuple.get_item(0)?;
            let actual_exponent = actual_tuple.get_item(1)?;
            let expected_mantissa = expected_tuple.get_item(0)?;
            let expected_exponent = expected_tuple.get_item(1)?;

            assert_eq!(
                actual_mantissa
                    .call_method0("tolist")?
                    .extract::<Vec<f64>>()?,
                expected_mantissa
                    .call_method0("tolist")?
                    .extract::<Vec<f64>>()?
            );
            assert_eq!(
                actual_exponent
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?,
                expected_exponent
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?
            );
            assert_eq!(
                actual_exponent
                    .call_method0("tolist")?
                    .extract::<Vec<i32>>()?,
                expected_exponent
                    .call_method0("tolist")?
                    .extract::<Vec<i32>>()?
            );
            Ok(())
        });
    }

    #[test]
    fn frexp_matches_numpy_special_values_and_shape() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let x = numeric_array(
                py,
                vec![vec![0.0, -0.0], vec![f64::INFINITY, f64::NAN]],
                "float64",
            );
            let actual = frexp(py, x.clone().unbind())?;

            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("frexp")?.call1((x,))?;

            let actual_tuple = actual.bind(py).downcast::<PyTuple>()?;
            let expected_tuple = expected.downcast::<PyTuple>()?;
            let actual_mantissa = actual_tuple.get_item(0)?;
            let actual_exponent = actual_tuple.get_item(1)?;
            let expected_mantissa = expected_tuple.get_item(0)?;
            let expected_exponent = expected_tuple.get_item(1)?;

            assert_eq!(
                actual_mantissa.getattr("shape")?.extract::<Vec<usize>>()?,
                vec![2, 2]
            );
            assert_eq!(
                actual_exponent.getattr("shape")?.extract::<Vec<usize>>()?,
                vec![2, 2]
            );
            assert_eq!(
                repr_string(&actual_mantissa.call_method0("tolist")?),
                repr_string(&expected_mantissa.call_method0("tolist")?)
            );
            assert_eq!(
                actual_exponent
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?,
                "int32"
            );
            assert_eq!(
                actual_exponent
                    .call_method0("tolist")?
                    .extract::<Vec<Vec<i32>>>()?,
                expected_exponent
                    .call_method0("tolist")?
                    .extract::<Vec<Vec<i32>>>()?
            );
            Ok(())
        });
    }

    #[test]
    fn modf_matches_numpy_basic() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let x = numeric_array(py, vec![0.0, -0.0, 1.5, -2.5], "float64");
            let actual = modf(py, x.clone().unbind())?;

            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("modf")?.call1((x,))?;

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
                    actual_item.call_method0("tolist")?.extract::<Vec<f64>>()?,
                    expected_item
                        .call_method0("tolist")?
                        .extract::<Vec<f64>>()?
                );
            }
            Ok(())
        });
    }

    #[test]
    fn modf_matches_numpy_special_values_and_shape() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let x = numeric_array(
                py,
                vec![vec![f64::INFINITY, f64::NEG_INFINITY], vec![f64::NAN, -0.0]],
                "float64",
            );
            let actual = modf(py, x.clone().unbind())?;

            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("modf")?.call1((x,))?;

            let actual_tuple = actual.bind(py).downcast::<PyTuple>()?;
            let expected_tuple = expected.downcast::<PyTuple>()?;
            let actual_fractional = actual_tuple.get_item(0)?;
            let actual_integral = actual_tuple.get_item(1)?;
            let expected_fractional = expected_tuple.get_item(0)?;
            let expected_integral = expected_tuple.get_item(1)?;

            assert_eq!(
                actual_fractional
                    .getattr("shape")?
                    .extract::<Vec<usize>>()?,
                vec![2, 2]
            );
            assert_eq!(
                actual_integral.getattr("shape")?.extract::<Vec<usize>>()?,
                vec![2, 2]
            );
            assert_eq!(
                repr_string(&actual_fractional.call_method0("tolist")?),
                repr_string(&expected_fractional.call_method0("tolist")?)
            );
            assert_eq!(
                repr_string(&actual_integral.call_method0("tolist")?),
                repr_string(&expected_integral.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn nan_to_num_matches_numpy_defaults() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let x = numeric_array(
                py,
                vec![0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
                "float64",
            );
            let actual = nan_to_num(py, x.clone().unbind(), 0.0, None, None)?;

            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("nan_to_num")?.call1((x,))?;

            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn nan_to_num_matches_numpy_custom_replacements_and_int_passthrough() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let float_values = numeric_array(
                py,
                vec![vec![f64::NAN, f64::INFINITY], vec![f64::NEG_INFINITY, 5.0]],
                "float64",
            );
            let actual_float = nan_to_num(
                py,
                float_values.clone().unbind(),
                1.5,
                Some(9.0),
                Some(-7.0),
            )?;

            let numpy = py.import("numpy")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("nan", 1.5)?;
            kwargs.set_item("posinf", 9.0)?;
            kwargs.set_item("neginf", -7.0)?;
            let expected_float = numpy
                .getattr("nan_to_num")?
                .call((float_values,), Some(&kwargs))?;

            assert_eq!(
                actual_float
                    .bind(py)
                    .getattr("shape")?
                    .extract::<Vec<usize>>()?,
                vec![2, 2]
            );
            assert_eq!(
                repr_string(&actual_float.bind(py).call_method0("tolist")?),
                repr_string(&expected_float.call_method0("tolist")?)
            );

            let int_values = numeric_array(py, vec![1_i64, 2_i64, 3_i64], "int64");
            let actual_int = nan_to_num(py, int_values.clone().unbind(), 0.0, None, None)?;
            let expected_int = numpy.getattr("nan_to_num")?.call1((int_values,))?;

            assert_eq!(
                actual_int
                    .bind(py)
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?,
                expected_int.getattr("dtype")?.str()?.extract::<String>()?
            );
            assert_eq!(
                actual_int
                    .bind(py)
                    .call_method0("tolist")?
                    .extract::<Vec<i64>>()?,
                expected_int.call_method0("tolist")?.extract::<Vec<i64>>()?
            );
            Ok(())
        });
    }

    #[test]
    fn take_matches_numpy_flat_multidimensional_indices() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let arr = numeric_array(
                py,
                vec![vec![10_i64, 20_i64, 30_i64], vec![40_i64, 50_i64, 60_i64]],
                "int64",
            );
            let indices = numeric_array(py, vec![vec![2_i64, 0_i64], vec![1_i64, 1_i64]], "int64");
            let actual = take(py, arr.clone().unbind(), indices.clone().unbind(), None)?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("take", (arr, indices))?;

            assert_eq!(
                actual.bind(py).getattr("shape")?.extract::<Vec<usize>>()?,
                expected.getattr("shape")?.extract::<Vec<usize>>()?
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn take_matches_numpy_axis_scalar_index() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let arr = numeric_array(
                py,
                vec![vec![10_i64, 20_i64, 30_i64], vec![40_i64, 50_i64, 60_i64]],
                "int64",
            );
            let index = 1_i64.into_pyobject(py)?.unbind();
            let actual = take(
                py,
                arr.clone().unbind(),
                index.clone_ref(py).into(),
                Some(1),
            )?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method(
                "take",
                (arr, index.bind(py)),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("axis", 1)?;
                    kwargs
                }),
            )?;

            assert_eq!(
                actual.bind(py).getattr("shape")?.extract::<Vec<usize>>()?,
                expected.getattr("shape")?.extract::<Vec<usize>>()?
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn take_matches_numpy_axis_multidimensional_indices() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let arr = numeric_array(
                py,
                vec![vec![10_i64, 20_i64, 30_i64], vec![40_i64, 50_i64, 60_i64]],
                "int64",
            );
            let indices = numeric_array(py, vec![vec![2_i64, 0_i64], vec![1_i64, 1_i64]], "int64");
            let actual = take(py, arr.clone().unbind(), indices.clone().unbind(), Some(1))?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method(
                "take",
                (arr, indices),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("axis", 1)?;
                    kwargs
                }),
            )?;

            assert_eq!(
                actual.bind(py).getattr("shape")?.extract::<Vec<usize>>()?,
                expected.getattr("shape")?.extract::<Vec<usize>>()?
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn take_preserves_large_uint64_values() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let large = (1_u64 << 63) + 5;
            let arr = numeric_array(py, vec![large, 7_u64, 9_u64], "uint64");
            let indices = numeric_array(py, vec![0_i64, 2_i64], "int64");
            let actual = take(py, arr.clone().unbind(), indices.clone().unbind(), None)?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("take", (arr, indices))?;

            assert_eq!(
                actual
                    .bind(py)
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?,
                expected.getattr("dtype")?.str()?.extract::<String>()?
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn put_matches_numpy_in_place_and_returns_none() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let arr = numeric_array(py, vec![1_i64, 2_i64, 3_i64, 4_i64], "int64");
            let expected = numeric_array(py, vec![1_i64, 2_i64, 3_i64, 4_i64], "int64");
            let indices = numeric_array(py, vec![3_i64, 1_i64, 1_i64], "int64");
            let values = numeric_array(py, vec![9_i64, 8_i64], "int64");

            let actual = put(
                py,
                arr.clone().unbind(),
                indices.clone().unbind(),
                values.clone().unbind(),
            )?;
            assert!(actual.bind(py).is_none());

            let numpy = py.import("numpy")?;
            numpy.call_method1("put", (expected.clone(), indices, values))?;

            assert_eq!(
                repr_string(&arr.call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn put_preserves_large_uint64_values_in_place() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let large = (1_u64 << 63) + 4099;
            let arr = numeric_array(py, vec![1_u64, 2_u64, 3_u64], "uint64");
            let expected = numeric_array(py, vec![1_u64, 2_u64, 3_u64], "uint64");
            let indices = numeric_array(py, vec![1_i64, 2_i64], "int64");
            let values = numeric_array(py, vec![large, large - 1], "uint64");

            let actual = put(
                py,
                arr.clone().unbind(),
                indices.clone().unbind(),
                values.clone().unbind(),
            )?;
            assert!(actual.bind(py).is_none());

            let numpy = py.import("numpy")?;
            numpy.call_method1("put", (expected.clone(), indices, values))?;

            assert_eq!(
                arr.getattr("dtype")?.str()?.extract::<String>()?,
                expected.getattr("dtype")?.str()?.extract::<String>()?
            );
            assert_eq!(
                repr_string(&arr.call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn place_matches_numpy_in_place_and_returns_none() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let arr = numeric_array(py, vec![1_i64, 2_i64, 3_i64, 4_i64], "int64");
            let expected = numeric_array(py, vec![1_i64, 2_i64, 3_i64, 4_i64], "int64");
            let mask = numeric_array(py, vec![0_i64, 1_i64, 1_i64, 0_i64], "int64");
            let values = numeric_array(py, vec![9_i64], "int64");

            let actual = place(
                py,
                arr.clone().unbind(),
                mask.clone().unbind(),
                values.clone().unbind(),
            )?;
            assert!(actual.bind(py).is_none());

            let numpy = py.import("numpy")?;
            numpy.call_method1("place", (expected.clone(), mask, values))?;

            assert_eq!(
                repr_string(&arr.call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn place_preserves_large_uint64_values_in_place() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let large = (1_u64 << 63) + 8195;
            let arr = numeric_array(py, vec![1_u64, 2_u64, 3_u64], "uint64");
            let expected = numeric_array(py, vec![1_u64, 2_u64, 3_u64], "uint64");
            let mask = numeric_array(py, vec![0_i64, 1_i64, 1_i64], "int64");
            let values = numeric_array(py, vec![large, large - 1], "uint64");

            let actual = place(
                py,
                arr.clone().unbind(),
                mask.clone().unbind(),
                values.clone().unbind(),
            )?;
            assert!(actual.bind(py).is_none());

            let numpy = py.import("numpy")?;
            numpy.call_method1("place", (expected.clone(), mask, values))?;

            assert_eq!(
                arr.getattr("dtype")?.str()?.extract::<String>()?,
                expected.getattr("dtype")?.str()?.extract::<String>()?
            );
            assert_eq!(
                repr_string(&arr.call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn putmask_matches_numpy_in_place_and_cycles_values() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let arr = numeric_array(py, vec![0_i64, 1_i64, 2_i64, 3_i64, 4_i64], "int64");
            let expected = numeric_array(py, vec![0_i64, 1_i64, 2_i64, 3_i64, 4_i64], "int64");
            let mask = numeric_array(py, vec![0_i64, 0_i64, 1_i64, 1_i64, 1_i64], "int64");
            let values = numeric_array(py, vec![-33_i64, -44_i64], "int64");

            let actual = putmask(
                py,
                arr.clone().unbind(),
                mask.clone().unbind(),
                values.clone().unbind(),
            )?;
            assert!(actual.bind(py).is_none());

            let numpy = py.import("numpy")?;
            numpy.call_method1("putmask", (expected.clone(), mask, values))?;

            assert_eq!(
                repr_string(&arr.call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn putmask_preserves_large_uint64_values_in_place() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let large = (1_u64 << 63) + 32_771;
            let arr = numeric_array(py, vec![1_u64, 2_u64, 3_u64], "uint64");
            let expected = numeric_array(py, vec![1_u64, 2_u64, 3_u64], "uint64");
            let mask = numeric_array(py, vec![0_i64, 1_i64, 1_i64], "int64");
            let values = numeric_array(py, vec![large, large - 1], "uint64");

            let actual = putmask(
                py,
                arr.clone().unbind(),
                mask.clone().unbind(),
                values.clone().unbind(),
            )?;
            assert!(actual.bind(py).is_none());

            let numpy = py.import("numpy")?;
            numpy.call_method1("putmask", (expected.clone(), mask, values))?;

            assert_eq!(
                arr.getattr("dtype")?.str()?.extract::<String>()?,
                expected.getattr("dtype")?.str()?.extract::<String>()?
            );
            assert_eq!(
                repr_string(&arr.call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn putmask_matches_numpy_with_overlapping_view_inputs() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let builtins = py.import("builtins")?;
            let slice = builtins.getattr("slice")?;

            let arr = numeric_array(py, vec![10_i64, 20_i64, 30_i64, 40_i64], "int64");
            let expected = numeric_array(py, vec![10_i64, 20_i64, 30_i64, 40_i64], "int64");

            let actual_view = arr.get_item(slice.call1((1, 4))?)?;
            let actual_values = arr.get_item(slice.call1((0, 3))?)?;
            let expected_view = expected.get_item(slice.call1((1, 4))?)?;
            let expected_values = expected.get_item(slice.call1((0, 3))?)?;
            let mask = numeric_array(py, vec![1_i64, 1_i64, 1_i64], "int64");

            let actual = putmask(
                py,
                actual_view.unbind(),
                mask.clone().unbind(),
                actual_values.unbind(),
            )?;
            assert!(actual.bind(py).is_none());

            let numpy = py.import("numpy")?;
            numpy.call_method1("putmask", (expected_view, mask, expected_values))?;

            assert_eq!(
                repr_string(&arr.call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn indices_matches_numpy_default_and_explicit_dtype() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let actual_default = indices(py, vec![2, 3], None)?;
            let numpy = py.import("numpy")?;
            let expected_default = numpy.call_method1("indices", ((2, 3),))?;
            assert_array_matches_numpy(actual_default.bind(py), &expected_default)?;

            let int32 = numpy.getattr("int32")?;
            let actual_typed = indices(py, vec![2, 3], Some(int32.clone().unbind()))?;
            let expected_typed = numpy.call_method1("indices", ((2, 3), int32))?;
            assert_array_matches_numpy(actual_typed.bind(py), &expected_typed)?;
            Ok(())
        });
    }

    #[test]
    fn diag_matches_numpy_for_vector_matrix_and_large_uint64_inputs() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let vector = numeric_array(py, vec![10_i64, 20_i64, 30_i64], "int64");
            let actual_vector = diag(py, vector.clone().unbind(), 1)?;
            let numpy = py.import("numpy")?;
            let expected_vector = numpy.call_method1("diag", (vector, 1))?;
            assert_array_matches_numpy(actual_vector.bind(py), &expected_vector)?;

            let matrix = numeric_array(
                py,
                vec![vec![1_i64, 2_i64, 3_i64], vec![4_i64, 5_i64, 6_i64]],
                "int64",
            );
            let actual_matrix = diag(py, matrix.clone().unbind(), -1)?;
            let expected_matrix = numpy.call_method1("diag", (matrix, -1))?;
            assert_array_matches_numpy(actual_matrix.bind(py), &expected_matrix)?;

            let large = (1_u64 << 63) + 777;
            let uint_vector = numeric_array(py, vec![large, large - 1], "uint64");
            let actual_uint = diag(py, uint_vector.clone().unbind(), 0)?;
            let expected_uint = numpy.call_method1("diag", (uint_vector, 0))?;
            assert_array_matches_numpy(actual_uint.bind(py), &expected_uint)?;
            Ok(())
        });
    }

    #[test]
    fn diagflat_matches_numpy_and_preserves_large_uint64_values() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let input = numeric_array(py, vec![vec![1_i64, 2_i64], vec![3_i64, 4_i64]], "int64");
            let actual = diagflat(py, input.clone().unbind(), 1)?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("diagflat", (input, 1))?;
            assert_array_matches_numpy(actual.bind(py), &expected)?;

            let large = (1_u64 << 63) + 65_539;
            let uint_input = numeric_array(py, vec![large, large - 1], "uint64");
            let actual_uint = diagflat(py, uint_input.clone().unbind(), 0)?;
            let expected_uint = numpy.call_method1("diagflat", (uint_input, 0))?;
            assert_array_matches_numpy(actual_uint.bind(py), &expected_uint)?;
            Ok(())
        });
    }

    #[test]
    fn diagonal_matches_numpy_for_offsets_axes_and_errors() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let array = numeric_array(
                py,
                vec![
                    vec![vec![0_i64, 1_i64], vec![2_i64, 3_i64], vec![4_i64, 5_i64]],
                    vec![vec![6_i64, 7_i64], vec![8_i64, 9_i64], vec![10_i64, 11_i64]],
                ],
                "int64",
            );
            let actual = diagonal(py, array.clone().unbind(), 0, 0, 2)?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method(
                "diagonal",
                (array,),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("offset", 0)?;
                    kwargs.set_item("axis1", 0)?;
                    kwargs.set_item("axis2", 2)?;
                    kwargs
                }),
            )?;
            assert_array_matches_numpy(actual.bind(py), &expected)?;

            let actual_neg_axes = diagonal(py, expected.clone().unbind(), 0, -2, -1)?;
            let expected_neg_axes = numpy.call_method(
                "diagonal",
                (expected.clone(),),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("offset", 0)?;
                    kwargs.set_item("axis1", -2)?;
                    kwargs.set_item("axis2", -1)?;
                    kwargs
                }),
            )?;
            assert_array_matches_numpy(actual_neg_axes.bind(py), &expected_neg_axes)?;

            let square = numeric_array(py, vec![vec![1_i64, 2_i64], vec![3_i64, 4_i64]], "int64");
            let err = diagonal(py, square.unbind(), 0, 0, 0).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(
                err.to_string()
                    .contains("axis1 and axis2 must be different")
            );
            Ok(())
        });
    }

    #[test]
    fn fill_diagonal_matches_numpy_for_wrap_nd_and_sequence_values() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let arr = numeric_array(
                py,
                vec![
                    vec![0_i64, 0_i64, 0_i64],
                    vec![0_i64, 0_i64, 0_i64],
                    vec![0_i64, 0_i64, 0_i64],
                    vec![0_i64, 0_i64, 0_i64],
                    vec![0_i64, 0_i64, 0_i64],
                ],
                "int64",
            );
            let expected = numeric_array(
                py,
                vec![
                    vec![0_i64, 0_i64, 0_i64],
                    vec![0_i64, 0_i64, 0_i64],
                    vec![0_i64, 0_i64, 0_i64],
                    vec![0_i64, 0_i64, 0_i64],
                    vec![0_i64, 0_i64, 0_i64],
                ],
                "int64",
            );
            let values = numeric_array(py, vec![9_i64, 8_i64], "int64");

            let actual = fill_diagonal(py, arr.clone().unbind(), values.clone().unbind(), true)?;
            assert!(actual.bind(py).is_none());

            let numpy = py.import("numpy")?;
            numpy.call_method(
                "fill_diagonal",
                (expected.clone(), values.clone()),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("wrap", true)?;
                    kwargs
                }),
            )?;
            assert_eq!(
                repr_string(&arr.call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );

            let cube = numeric_array(
                py,
                vec![
                    vec![vec![0_i64, 0_i64], vec![0_i64, 0_i64]],
                    vec![vec![0_i64, 0_i64], vec![0_i64, 0_i64]],
                ],
                "int64",
            );
            let expected_cube = numeric_array(
                py,
                vec![
                    vec![vec![0_i64, 0_i64], vec![0_i64, 0_i64]],
                    vec![vec![0_i64, 0_i64], vec![0_i64, 0_i64]],
                ],
                "int64",
            );

            let actual = fill_diagonal(
                py,
                cube.clone().unbind(),
                numeric_array(py, 4_i64, "int64").unbind(),
                false,
            )?;
            assert!(actual.bind(py).is_none());
            numpy.call_method1("fill_diagonal", (expected_cube.clone(), 4_i64))?;
            assert_eq!(
                repr_string(&cube.call_method0("tolist")?),
                repr_string(&expected_cube.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn fill_diagonal_rejects_low_dim_and_heterogeneous_nd_inputs() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let vector = numeric_array(py, vec![1_i64, 2_i64, 3_i64], "int64");
            let err = fill_diagonal(
                py,
                vector.unbind(),
                numeric_array(py, 5_i64, "int64").unbind(),
                false,
            )
            .unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err.to_string().contains("at least 2-d"));

            let non_uniform = numeric_array(
                py,
                vec![
                    vec![vec![0_i64, 0_i64, 0_i64], vec![0_i64, 0_i64, 0_i64]],
                    vec![vec![0_i64, 0_i64, 0_i64], vec![0_i64, 0_i64, 0_i64]],
                ],
                "int64",
            );
            let err = fill_diagonal(
                py,
                non_uniform.unbind(),
                numeric_array(py, 2_i64, "int64").unbind(),
                false,
            )
            .unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err.to_string().contains("equal length"));
            Ok(())
        });
    }

    #[test]
    fn ix_matches_numpy_for_multiple_arrays_bool_and_empty_inputs() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let lhs = numeric_array(py, vec![1_i64, 3_i64], "int64");
            let rhs = numeric_array(py, vec![2_i64, 5_i64], "int64");
            let args = PyTuple::new(py, [lhs.clone(), rhs.clone()])?;
            let actual = ix_(py, &args)?;
            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("ix_")?.call1((lhs, rhs))?;
            assert_index_tuple_matches_numpy(actual.bind(py), &expected)?;

            let bool_list = PyList::new(py, [true, false, true, true])?;
            let bool_args = PyTuple::new(py, [bool_list])?;
            let actual_bool = ix_(py, &bool_args)?;
            let expected_bool = numpy.getattr("ix_")?.call1((bool_args.get_item(0)?,))?;
            assert_index_tuple_matches_numpy(actual_bool.bind(py), &expected_bool)?;

            let empty_list = PyList::empty(py);
            let empty_args = PyTuple::new(py, [empty_list])?;
            let actual_empty = ix_(py, &empty_args)?;
            let expected_empty = numpy.getattr("ix_")?.call1((empty_args.get_item(0)?,))?;
            assert_index_tuple_matches_numpy(actual_empty.bind(py), &expected_empty)?;

            let empty_float = numeric_array(py, Vec::<f32>::new(), "float32");
            let empty_float_args = PyTuple::new(py, [empty_float.clone()])?;
            let actual_float = ix_(py, &empty_float_args)?;
            let expected_float = numpy.getattr("ix_")?.call1((empty_float,))?;
            assert_index_tuple_matches_numpy(actual_float.bind(py), &expected_float)?;
            Ok(())
        });
    }

    #[test]
    fn ix_rejects_non_1d_inputs() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let matrix = numeric_array(
                py,
                vec![vec![1_i64, 2_i64, 3_i64], vec![4_i64, 5_i64, 6_i64]],
                "int64",
            );
            let args = PyTuple::new(py, [matrix])?;
            let err = ix_(py, &args).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(
                err.to_string()
                    .contains("Cross index must be 1 dimensional")
            );
            Ok(())
        });
    }

    #[test]
    fn meshgrid_matches_numpy_for_dense_default_and_ij_indexing() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let x = numeric_array(py, vec![1_f32, 2_f32, 3_f32], "float32");
            let large = (1_u64 << 63) + 19;
            let y = numeric_array(py, vec![large, large - 1], "uint64");
            let args = PyTuple::new(py, [x.clone(), y.clone()])?;

            let actual = meshgrid(py, &args, true, false, "xy")?;
            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("meshgrid")?.call1((x.clone(), y.clone()))?;
            assert_index_tuple_matches_numpy(actual.bind(py), &expected)?;

            let actual_ij = meshgrid(py, &args, true, false, "ij")?;
            let expected_ij = numpy.getattr("meshgrid")?.call(
                (x, y),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("indexing", "ij")?;
                    kwargs
                }),
            )?;
            assert_index_tuple_matches_numpy(actual_ij.bind(py), &expected_ij)?;
            Ok(())
        });
    }

    #[test]
    fn meshgrid_matches_numpy_for_empty_scalar_flattened_and_sparse_inputs() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let numpy = py.import("numpy")?;

            let empty_args = PyTuple::new(py, Vec::<i32>::new())?;
            let actual_empty = meshgrid(py, &empty_args, true, false, "xy")?;
            let expected_empty = numpy.getattr("meshgrid")?.call0()?;
            assert_eq!(
                repr_string(actual_empty.bind(py)),
                repr_string(&expected_empty)
            );

            let scalar = 5_i64.into_pyobject(py)?.into_any().unbind();
            let scalar_args = PyTuple::new(py, [scalar.clone_ref(py)])?;
            let actual_scalar = meshgrid(py, &scalar_args, true, false, "xy")?;
            let expected_scalar = numpy.getattr("meshgrid")?.call1((scalar.bind(py),))?;
            assert_index_tuple_matches_numpy(actual_scalar.bind(py), &expected_scalar)?;

            let matrix = numeric_array(py, vec![vec![1_i64], vec![2_i64]], "int64");
            let y = numeric_array(py, vec![10_i64, 20_i64], "int64");
            let flat_args = PyTuple::new(py, [matrix.clone(), y.clone()])?;

            let actual_flat = meshgrid(py, &flat_args, true, false, "xy")?;
            let expected_flat = numpy
                .getattr("meshgrid")?
                .call1((matrix.clone(), y.clone()))?;
            assert_index_tuple_matches_numpy(actual_flat.bind(py), &expected_flat)?;

            let actual_sparse = meshgrid(py, &flat_args, true, true, "xy")?;
            let expected_sparse = numpy.getattr("meshgrid")?.call(
                (matrix, y),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("sparse", true)?;
                    kwargs
                }),
            )?;
            assert_index_tuple_matches_numpy(actual_sparse.bind(py), &expected_sparse)?;

            assert!(meshgrid(py, &flat_args, true, false, "bad").is_err());
            Ok(())
        });
    }

    #[test]
    fn meshgrid_matches_numpy_for_copy_false_writeback_and_object_sparse() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let x = numeric_array(py, vec![1_i64, 2_i64], "int64");
            let y = numeric_array(py, vec![3_i64, 4_i64, 5_i64], "int64");
            let expected_x = numeric_array(py, vec![1_i64, 2_i64], "int64");
            let expected_y = numeric_array(py, vec![3_i64, 4_i64, 5_i64], "int64");
            let args = PyTuple::new(py, [x.clone(), y.clone()])?;

            let actual = meshgrid(py, &args, false, false, "xy")?;
            let numpy = py.import("numpy")?;
            let expected = numpy.getattr("meshgrid")?.call(
                (expected_x.clone(), expected_y.clone()),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("copy", false)?;
                    kwargs
                }),
            )?;
            assert_index_tuple_matches_numpy(actual.bind(py), &expected)?;

            let actual_tuple = actual.bind(py).downcast::<PyTuple>()?;
            let expected_tuple = expected.downcast::<PyTuple>()?;
            let actual_x_grid = actual_tuple.get_item(0)?;
            let actual_y_grid = actual_tuple.get_item(1)?;
            let expected_x_grid = expected_tuple.get_item(0)?;
            let expected_y_grid = expected_tuple.get_item(1)?;

            assert!(
                !actual_x_grid
                    .getattr("flags")?
                    .get_item("OWNDATA")?
                    .extract::<bool>()?
            );
            assert!(
                actual_x_grid
                    .getattr("flags")?
                    .get_item("WRITEABLE")?
                    .extract::<bool>()?
            );

            actual_x_grid.call_method1("fill", (99_i64,))?;
            expected_x_grid.call_method1("fill", (99_i64,))?;
            actual_y_grid.call_method1("fill", (-7_i64,))?;
            expected_y_grid.call_method1("fill", (-7_i64,))?;

            assert_eq!(
                repr_string(&x.call_method0("tolist")?),
                repr_string(&expected_x.call_method0("tolist")?)
            );
            assert_eq!(
                repr_string(&y.call_method0("tolist")?),
                repr_string(&expected_y.call_method0("tolist")?)
            );

            let object_x = object_array(py, vec!["left", "right"]);
            let object_y = object_array(py, vec!["north", "south", "west"]);
            let object_args = PyTuple::new(py, [object_x.clone(), object_y.clone()])?;
            let actual_object = meshgrid(py, &object_args, true, true, "xy")?;
            let expected_object = numpy.getattr("meshgrid")?.call(
                (object_x, object_y),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("copy", true)?;
                    kwargs.set_item("sparse", true)?;
                    kwargs
                }),
            )?;
            assert_index_tuple_matches_numpy(actual_object.bind(py), &expected_object)?;
            Ok(())
        });
    }

    #[test]
    fn mgrid_ogrid_match_numpy_for_single_and_multi_slice_shapes() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let numpy = py.import("numpy")?;
            let builtins = py.import("builtins")?;

            let complex_step = builtins.getattr("complex")?.call1((0_i64, 10_i64))?;
            let single_key = slice_object(
                py,
                Some((-1_i64).into_pyobject(py)?.into_any().unbind()),
                Some(1_i64.into_pyobject(py)?.into_any().unbind()),
                Some(complex_step.unbind()),
            )?;
            let actual_mgrid = module.getattr("mgrid")?.get_item(single_key.bind(py))?;
            let expected_mgrid = numpy.getattr("mgrid")?.get_item(single_key.bind(py))?;
            assert_array_matches_numpy(&actual_mgrid, &expected_mgrid)?;

            let actual_ogrid = module.getattr("ogrid")?.get_item(single_key.bind(py))?;
            let expected_ogrid = numpy.getattr("ogrid")?.get_item(single_key.bind(py))?;
            assert_array_matches_numpy(&actual_ogrid, &expected_ogrid)?;

            let row_key = slice_object(
                py,
                Some(0_i64.into_pyobject(py)?.into_any().unbind()),
                Some(4_i64.into_pyobject(py)?.into_any().unbind()),
                None,
            )?;
            let col_key = slice_object(
                py,
                Some(0_i64.into_pyobject(py)?.into_any().unbind()),
                Some(5_i64.into_pyobject(py)?.into_any().unbind()),
                None,
            )?;
            let dense_key = PyTuple::new(py, [row_key.bind(py), col_key.bind(py)])?;

            let actual_dense = module.getattr("mgrid")?.get_item(&dense_key)?;
            let expected_dense = numpy.getattr("mgrid")?.get_item(&dense_key)?;
            assert_array_matches_numpy(&actual_dense, &expected_dense)?;

            let actual_sparse = module.getattr("ogrid")?.get_item(&dense_key)?;
            let expected_sparse = numpy.getattr("ogrid")?.get_item(&dense_key)?;
            assert_index_tuple_matches_numpy(&actual_sparse, &expected_sparse)?;
            Ok(())
        });
    }

    #[test]
    fn mgrid_ogrid_match_numpy_for_none_defaults_and_tuple_single_slice_shapes() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let numpy = py.import("numpy")?;

            let default_start = slice_object(
                py,
                None,
                Some(3_i64.into_pyobject(py)?.into_any().unbind()),
                None,
            )?;
            let actual_mgrid = module.getattr("mgrid")?.get_item(default_start.bind(py))?;
            let expected_mgrid = numpy.getattr("mgrid")?.get_item(default_start.bind(py))?;
            assert_array_matches_numpy(&actual_mgrid, &expected_mgrid)?;

            let actual_ogrid = module.getattr("ogrid")?.get_item(default_start.bind(py))?;
            let expected_ogrid = numpy.getattr("ogrid")?.get_item(default_start.bind(py))?;
            assert_array_matches_numpy(&actual_ogrid, &expected_ogrid)?;

            let tuple_single_key = PyTuple::new(
                py,
                [slice_object(
                    py,
                    Some(0_i64.into_pyobject(py)?.into_any().unbind()),
                    Some(5_i64.into_pyobject(py)?.into_any().unbind()),
                    None,
                )?
                .bind(py)],
            )?;

            let actual_tuple_mgrid = module.getattr("mgrid")?.get_item(&tuple_single_key)?;
            let expected_tuple_mgrid = numpy.getattr("mgrid")?.get_item(&tuple_single_key)?;
            assert_array_matches_numpy(&actual_tuple_mgrid, &expected_tuple_mgrid)?;

            let actual_tuple_ogrid = module.getattr("ogrid")?.get_item(&tuple_single_key)?;
            let expected_tuple_ogrid = numpy.getattr("ogrid")?.get_item(&tuple_single_key)?;
            assert_index_tuple_matches_numpy(&actual_tuple_ogrid, &expected_tuple_ogrid)?;
            Ok(())
        });
    }

    #[test]
    fn mgrid_ogrid_match_numpy_for_float32_and_complex_count_dtypes() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let numpy = py.import("numpy")?;
            let builtins = py.import("builtins")?;

            let float32_key = slice_object(
                py,
                Some(numpy.getattr("float32")?.call1((0.1_f64,))?.unbind()),
                Some(numpy.getattr("float32")?.call1((0.33_f64,))?.unbind()),
                Some(numpy.getattr("float32")?.call1((0.1_f64,))?.unbind()),
            )?;
            let actual_float32 = module.getattr("mgrid")?.get_item(float32_key.bind(py))?;
            let expected_float32 = numpy.getattr("mgrid")?.get_item(float32_key.bind(py))?;
            assert_array_matches_numpy(&actual_float32, &expected_float32)?;

            let float32_tuple = PyTuple::new(py, [float32_key.bind(py)])?;
            let actual_float32_tuple = module.getattr("mgrid")?.get_item(&float32_tuple)?;
            let expected_float32_tuple = numpy.getattr("mgrid")?.get_item(&float32_tuple)?;
            assert_array_matches_numpy(&actual_float32_tuple, &expected_float32_tuple)?;

            let complex64_key = slice_object(
                py,
                Some(0.1_f64.into_pyobject(py)?.into_any().unbind()),
                Some(0.3_f64.into_pyobject(py)?.into_any().unbind()),
                Some(
                    numpy
                        .getattr("complex64")?
                        .call1((builtins.getattr("complex")?.call1((0_i64, 3_i64))?,))?
                        .unbind(),
                ),
            )?;
            let actual_complex = module.getattr("ogrid")?.get_item(complex64_key.bind(py))?;
            let expected_complex = numpy.getattr("ogrid")?.get_item(complex64_key.bind(py))?;
            assert_array_matches_numpy(&actual_complex, &expected_complex)?;
            Ok(())
        });
    }

    #[test]
    fn mgrid_ogrid_reject_non_slice_indexing() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;

            let err = module
                .getattr("mgrid")?
                .get_item(0_i64.into_pyobject(py)?.into_any())
                .unwrap_err();
            assert!(err.is_instance_of::<PyTypeError>(py));

            let bad_index = 0_i64.into_pyobject(py)?.into_any().unbind();
            let mixed_slice = slice_object(
                py,
                Some(0_i64.into_pyobject(py)?.into_any().unbind()),
                Some(4_i64.into_pyobject(py)?.into_any().unbind()),
                None,
            )?;
            let mixed_key = PyTuple::new(py, [bad_index.bind(py), mixed_slice.bind(py)])?;
            let err = module.getattr("ogrid")?.get_item(&mixed_key).unwrap_err();
            assert!(err.is_instance_of::<PyTypeError>(py));
            Ok(())
        });
    }

    #[test]
    fn r_c_match_numpy_for_numeric_scalars_and_slice_expansion() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let numpy = py.import("numpy")?;
            let builtins = py.import("builtins")?;

            let left = numeric_array(py, vec![1_i64, 2_i64, 3_i64], "int64").unbind();
            let right = numeric_array(py, vec![4_i64, 5_i64, 6_i64], "int64").unbind();
            let zero_a = 0_i64.into_pyobject(py)?.into_any().unbind();
            let zero_b = 0_i64.into_pyobject(py)?.into_any().unbind();
            let r_items = [left.clone_ref(py), zero_a, zero_b, right.clone_ref(py)];
            let r_key = PyTuple::new(py, r_items.iter().map(|item| item.bind(py)))?;
            let actual_r = module.getattr("r_")?.get_item(&r_key)?;
            let expected_r = numpy.getattr("r_")?.get_item(&r_key)?;
            assert_array_matches_numpy(&actual_r, &expected_r)?;

            let complex_step = builtins.getattr("complex")?.call1((0_i64, 8_i64))?;
            let slice_key = slice_object(
                py,
                Some((-1_i64).into_pyobject(py)?.into_any().unbind()),
                Some(1_i64.into_pyobject(py)?.into_any().unbind()),
                Some(complex_step.unbind()),
            )?;
            let actual_slice = module.getattr("r_")?.get_item(slice_key.bind(py))?;
            let expected_slice = numpy.getattr("r_")?.get_item(slice_key.bind(py))?;
            assert_array_matches_numpy(&actual_slice, &expected_slice)?;

            let c_left = numeric_array(py, vec![1_i64, 2_i64, 3_i64], "int64").unbind();
            let c_right = numeric_array(py, vec![4_i64, 5_i64, 6_i64], "int64").unbind();
            let c_key = PyTuple::new(py, [c_left.bind(py), c_right.bind(py)])?;
            let actual_c = module.getattr("c_")?.get_item(&c_key)?;
            let expected_c = numpy.getattr("c_")?.get_item(&c_key)?;
            assert_array_matches_numpy(&actual_c, &expected_c)?;

            let row_left = numeric_array(py, vec![vec![1_i64, 2_i64, 3_i64]], "int64").unbind();
            let row_right = numeric_array(py, vec![vec![4_i64, 5_i64, 6_i64]], "int64").unbind();
            let row_zero_a = 0_i64.into_pyobject(py)?.into_any().unbind();
            let row_zero_b = 0_i64.into_pyobject(py)?.into_any().unbind();
            let c_row_items = [
                row_left.clone_ref(py),
                row_zero_a,
                row_zero_b,
                row_right.clone_ref(py),
            ];
            let c_row_key = PyTuple::new(py, c_row_items.iter().map(|item| item.bind(py)))?;
            let actual_c_rows = module.getattr("c_")?.get_item(&c_row_key)?;
            let expected_c_rows = numpy.getattr("c_")?.get_item(&c_row_key)?;
            assert_array_matches_numpy(&actual_c_rows, &expected_c_rows)?;
            Ok(())
        });
    }

    #[test]
    fn r_c_match_numpy_for_directives_and_object_dtype_fallbacks() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let numpy = py.import("numpy")?;

            let matrix_left =
                numeric_array(py, vec![vec![0_i64, 1_i64], vec![2_i64, 3_i64]], "int64").unbind();
            let matrix_right =
                numeric_array(py, vec![vec![4_i64, 5_i64], vec![6_i64, 7_i64]], "int64").unbind();
            let directive = "1".into_pyobject(py)?.into_any().unbind();
            let directive_items = [
                directive,
                matrix_left.clone_ref(py),
                matrix_right.clone_ref(py),
            ];
            let directive_key = PyTuple::new(py, directive_items.iter().map(|item| item.bind(py)))?;
            let actual_directive = module.getattr("r_")?.get_item(&directive_key)?;
            let expected_directive = numpy.getattr("r_")?.get_item(&directive_key)?;
            assert_array_matches_numpy(&actual_directive, &expected_directive)?;

            let matrix_mode = "r".into_pyobject(py)?.into_any().unbind();
            let matrix_a = numeric_array(py, vec![1_i64, 2_i64, 3_i64], "int64").unbind();
            let matrix_b = numeric_array(py, vec![4_i64, 5_i64, 6_i64], "int64").unbind();
            let matrix_items = [matrix_mode, matrix_a, matrix_b];
            let matrix_key = PyTuple::new(py, matrix_items.iter().map(|item| item.bind(py)))?;
            let actual_matrix = module.getattr("r_")?.get_item(&matrix_key)?;
            let expected_matrix = numpy.getattr("r_")?.get_item(&matrix_key)?;
            assert_array_matches_numpy(&actual_matrix, &expected_matrix)?;

            let object_left = object_array(py, vec!["north", "south"]).unbind();
            let object_right = object_array(py, vec!["east", "west"]).unbind();
            let object_key = PyTuple::new(py, [object_left.bind(py), object_right.bind(py)])?;
            let actual_object = module.getattr("c_")?.get_item(&object_key)?;
            let expected_object = numpy.getattr("c_")?.get_item(&object_key)?;
            assert_array_matches_numpy(&actual_object, &expected_object)?;
            Ok(())
        });
    }

    #[test]
    fn stack_helpers_match_numpy_for_scalar_and_numeric_shapes() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let numpy = py.import("numpy")?;

            let scalar_a = numpy.getattr("array")?.call1((1_i64,))?;
            let scalar_b = numpy.getattr("array")?.call1((2_i64,))?;
            let scalar_seq = PyList::new(py, [scalar_a.clone(), scalar_b.clone()])?;

            let actual_v = module.getattr("vstack")?.call1((scalar_seq.clone(),))?;
            let expected_v = numpy.getattr("vstack")?.call1((scalar_seq.clone(),))?;
            assert_array_matches_numpy(&actual_v, &expected_v)?;

            let actual_h = module.getattr("hstack")?.call1((scalar_seq.clone(),))?;
            let expected_h = numpy.getattr("hstack")?.call1((scalar_seq.clone(),))?;
            assert_array_matches_numpy(&actual_h, &expected_h)?;

            let actual_d = module.getattr("dstack")?.call1((scalar_seq.clone(),))?;
            let expected_d = numpy.getattr("dstack")?.call1((scalar_seq.clone(),))?;
            assert_array_matches_numpy(&actual_d, &expected_d)?;

            let actual_column = module.getattr("column_stack")?.call1((scalar_seq,))?;
            let expected_column = numpy
                .getattr("column_stack")?
                .call1((PyList::new(py, [scalar_a.clone(), scalar_b.clone()])?,))?;
            assert_array_matches_numpy(&actual_column, &expected_column)?;

            let left = numeric_array(py, vec![1_i64, 2_i64, 3_i64], "int64");
            let right = numeric_array(py, vec![4_i64, 5_i64, 6_i64], "int64");
            let seq = PyTuple::new(py, [left.clone(), right.clone()])?;

            let actual_column_vec = module.getattr("column_stack")?.call1((seq.clone(),))?;
            let expected_column_vec = numpy.getattr("column_stack")?.call1((seq.clone(),))?;
            assert_array_matches_numpy(&actual_column_vec, &expected_column_vec)?;

            let actual_d_vec = module.getattr("dstack")?.call1((seq.clone(),))?;
            let expected_d_vec = numpy.getattr("dstack")?.call1((seq,))?;
            assert_array_matches_numpy(&actual_d_vec, &expected_d_vec)?;
            Ok(())
        });
    }

    #[test]
    fn vstack_hstack_match_numpy_for_dtype_and_sequence_rules() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let numpy = py.import("numpy")?;
            let builtins = py.import("builtins")?;

            let a = numeric_array(py, vec![1_i64, 2_i64, 3_i64], "int64");
            let b = numeric_array(py, vec![2.5_f64, 3.5_f64, 4.5_f64], "float64");
            let seq = PyTuple::new(py, [a.clone(), b.clone()])?;

            let actual_v = module.getattr("vstack")?.call(
                (seq.clone(),),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("dtype", numpy.getattr("int64")?)?;
                    kwargs.set_item("casting", "unsafe")?;
                    kwargs
                }),
            )?;
            let expected_v = numpy.getattr("vstack")?.call(
                (seq.clone(),),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("dtype", numpy.getattr("int64")?)?;
                    kwargs.set_item("casting", "unsafe")?;
                    kwargs
                }),
            )?;
            assert_array_matches_numpy(&actual_v, &expected_v)?;

            let actual_h = module.getattr("hstack")?.call(
                (seq.clone(),),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("dtype", numpy.getattr("int64")?)?;
                    kwargs.set_item("casting", "unsafe")?;
                    kwargs
                }),
            )?;
            let expected_h = numpy.getattr("hstack")?.call(
                (seq.clone(),),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("dtype", numpy.getattr("int64")?)?;
                    kwargs.set_item("casting", "unsafe")?;
                    kwargs
                }),
            )?;
            assert_array_matches_numpy(&actual_h, &expected_h)?;

            let err = module
                .getattr("vstack")?
                .call(
                    (seq.clone(),),
                    Some(&{
                        let kwargs = PyDict::new(py);
                        kwargs.set_item("dtype", numpy.getattr("int64")?)?;
                        kwargs.set_item("casting", "safe")?;
                        kwargs
                    }),
                )
                .unwrap_err();
            assert!(err.is_instance_of::<PyTypeError>(py));

            let iter_source = PyList::new(py, [a.clone(), b.clone()])?;
            let iterator = builtins.getattr("iter")?.call1((iter_source,))?;
            let err = module.getattr("hstack")?.call1((iterator,)).unwrap_err();
            assert!(err.is_instance_of::<PyTypeError>(py));
            assert!(
                err.to_string()
                    .contains("arrays to stack must be passed as a \"sequence\"")
            );
            Ok(())
        });
    }

    #[test]
    fn dstack_column_stack_match_numpy_for_object_fallbacks() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let numpy = py.import("numpy")?;

            let left = object_array(py, vec!["north", "south"]);
            let right = object_array(py, vec!["east", "west"]);
            let seq = PyTuple::new(py, [left.clone(), right.clone()])?;

            let actual_column = module.getattr("column_stack")?.call1((seq.clone(),))?;
            let expected_column = numpy.getattr("column_stack")?.call1((seq.clone(),))?;
            assert_array_matches_numpy(&actual_column, &expected_column)?;

            let actual_d = module.getattr("dstack")?.call1((seq.clone(),))?;
            let expected_d = numpy.getattr("dstack")?.call1((seq,))?;
            assert_array_matches_numpy(&actual_d, &expected_d)?;
            Ok(())
        });
    }

    #[test]
    fn split_family_match_numpy_for_integer_sections_and_shapes() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let numpy = py.import("numpy")?;

            let one_d = numeric_array(py, vec![0_i64, 1_i64, 2_i64, 3_i64, 4_i64, 5_i64], "int64");
            let actual_split = module.getattr("split")?.call1((one_d.clone(), 3_i64))?;
            let expected_split = numpy.getattr("split")?.call1((one_d.clone(), 3_i64))?;
            assert_array_list_matches_numpy(&actual_split, &expected_split)?;

            let actual_array_split = module
                .getattr("array_split")?
                .call1((one_d.clone(), 4_i64))?;
            let expected_array_split = numpy
                .getattr("array_split")?
                .call1((one_d.clone(), 4_i64))?;
            assert_array_list_matches_numpy(&actual_array_split, &expected_array_split)?;

            let actual_array_split_float = module
                .getattr("array_split")?
                .call1((one_d.clone(), 3.5_f64))?;
            let expected_array_split_float = numpy
                .getattr("array_split")?
                .call1((one_d.clone(), 3.5_f64))?;
            assert_array_list_matches_numpy(
                &actual_array_split_float,
                &expected_array_split_float,
            )?;

            let actual_hsplit = module.getattr("hsplit")?.call1((one_d.clone(), 2_i64))?;
            let expected_hsplit = numpy.getattr("hsplit")?.call1((one_d.clone(), 2_i64))?;
            assert_array_list_matches_numpy(&actual_hsplit, &expected_hsplit)?;

            let matrix = numeric_array(
                py,
                vec![
                    vec![1_i64, 2_i64, 3_i64, 4_i64],
                    vec![5_i64, 6_i64, 7_i64, 8_i64],
                ],
                "int64",
            );
            let actual_vsplit = module.getattr("vsplit")?.call1((matrix.clone(), 2_i64))?;
            let expected_vsplit = numpy.getattr("vsplit")?.call1((matrix.clone(), 2_i64))?;
            assert_array_list_matches_numpy(&actual_vsplit, &expected_vsplit)?;

            let cube = numpy
                .getattr("arange")?
                .call1((8_i64,))?
                .call_method1("reshape", ((2_i64, 2_i64, 2_i64),))?;
            let actual_dsplit = module.getattr("dsplit")?.call1((cube.clone(), 2_i64))?;
            let expected_dsplit = numpy.getattr("dsplit")?.call1((cube.clone(), 2_i64))?;
            assert_array_list_matches_numpy(&actual_dsplit, &expected_dsplit)?;
            Ok(())
        });
    }

    #[test]
    fn split_family_match_numpy_for_index_fallbacks_and_object_arrays() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let numpy = py.import("numpy")?;

            let one_d = numeric_array(py, vec![0_i64, 1_i64, 2_i64, 3_i64, 4_i64, 5_i64], "int64");
            let cut_points = PyList::new(py, [1_i64, 4_i64])?;
            let actual_split = module
                .getattr("split")?
                .call1((one_d.clone(), cut_points.clone()))?;
            let expected_split = numpy
                .getattr("split")?
                .call1((one_d.clone(), cut_points.clone()))?;
            assert_array_list_matches_numpy(&actual_split, &expected_split)?;

            let negative_points = PyList::new(py, [-1_i64, 2_i64])?;
            let actual_array_split = module
                .getattr("array_split")?
                .call1((one_d.clone(), negative_points.clone()))?;
            let expected_array_split = numpy
                .getattr("array_split")?
                .call1((one_d.clone(), negative_points.clone()))?;
            assert_array_list_matches_numpy(&actual_array_split, &expected_array_split)?;

            let object_matrix = object_array(
                py,
                vec![
                    vec!["north", "south", "east", "west"],
                    vec!["n2", "s2", "e2", "w2"],
                ],
            );
            let object_points = PyList::new(py, [1_i64, 3_i64])?;
            let actual_hsplit = module
                .getattr("hsplit")?
                .call1((object_matrix.clone(), object_points.clone()))?;
            let expected_hsplit = numpy
                .getattr("hsplit")?
                .call1((object_matrix.clone(), object_points.clone()))?;
            assert_array_list_matches_numpy(&actual_hsplit, &expected_hsplit)?;
            Ok(())
        });
    }

    #[test]
    fn split_family_error_messages_match_numpy() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let numpy = py.import("numpy")?;

            let scalar = numpy.getattr("array")?.call1((1_i64,))?;
            let actual_hsplit_err = module
                .getattr("hsplit")?
                .call1((scalar.clone(), 1_i64))
                .unwrap_err();
            let expected_hsplit_err = numpy
                .getattr("hsplit")?
                .call1((scalar.clone(), 1_i64))
                .unwrap_err();
            assert_eq!(
                actual_hsplit_err.to_string(),
                expected_hsplit_err.to_string()
            );

            let one_d = numeric_array(py, vec![1_i64, 2_i64, 3_i64, 4_i64], "int64");
            let actual_vsplit_err = module
                .getattr("vsplit")?
                .call1((one_d.clone(), 2_i64))
                .unwrap_err();
            let expected_vsplit_err = numpy
                .getattr("vsplit")?
                .call1((one_d.clone(), 2_i64))
                .unwrap_err();
            assert_eq!(
                actual_vsplit_err.to_string(),
                expected_vsplit_err.to_string()
            );

            let two_d = numeric_array(
                py,
                vec![
                    vec![1_i64, 2_i64, 3_i64, 4_i64],
                    vec![5_i64, 6_i64, 7_i64, 8_i64],
                ],
                "int64",
            );
            let actual_dsplit_err = module
                .getattr("dsplit")?
                .call1((two_d.clone(), 2_i64))
                .unwrap_err();
            let expected_dsplit_err = numpy
                .getattr("dsplit")?
                .call1((two_d.clone(), 2_i64))
                .unwrap_err();
            assert_eq!(
                actual_dsplit_err.to_string(),
                expected_dsplit_err.to_string()
            );

            let actual_split_err = module
                .getattr("split")?
                .call1((one_d.clone(), 0_i64))
                .unwrap_err();
            let expected_split_err = numpy
                .getattr("split")?
                .call1((one_d.clone(), 0_i64))
                .unwrap_err();
            assert_eq!(actual_split_err.to_string(), expected_split_err.to_string());

            let actual_array_split_err = module
                .getattr("array_split")?
                .call1((one_d.clone(), 0_i64))
                .unwrap_err();
            let expected_array_split_err = numpy
                .getattr("array_split")?
                .call1((one_d.clone(), 0_i64))
                .unwrap_err();
            assert_eq!(
                actual_array_split_err.to_string(),
                expected_array_split_err.to_string()
            );
            Ok(())
        });
    }

    #[test]
    fn ravel_unravel_index_match_numpy_for_basic_order_and_mode_cases() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let numpy = py.import("numpy")?;
            let dims = PyTuple::new(py, [7_usize, 6_usize])?;
            let actual_scalar = unravel_index(
                py,
                2_i64.into_pyobject(py)?.into_any().unbind(),
                dims.clone().into_any().unbind(),
                "F",
            )?;
            let expected_scalar = numpy.getattr("unravel_index")?.call(
                (2_i64, dims.clone()),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("order", "F")?;
                    kwargs
                }),
            )?;
            assert_eq!(
                repr_string(actual_scalar.bind(py)),
                repr_string(&expected_scalar)
            );

            let flat = numeric_array(py, vec![31_i64, 41_i64, 13_i64], "int64");
            let actual_unravel = unravel_index(
                py,
                flat.clone().unbind(),
                dims.clone().into_any().unbind(),
                "F",
            )?;
            let expected_unravel = numpy.getattr("unravel_index")?.call(
                (flat, dims.clone()),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("order", "F")?;
                    kwargs
                }),
            )?;
            assert_index_tuple_matches_numpy(actual_unravel.bind(py), &expected_unravel)?;

            let coords = numeric_array(
                py,
                vec![vec![3_i64, 6_i64, 6_i64], vec![4_i64, 5_i64, 1_i64]],
                "int64",
            );
            let actual_ravel = ravel_multi_index(
                py,
                coords.clone().unbind(),
                dims.clone().into_any().unbind(),
                None,
                "F",
            )?;
            let expected_ravel = numpy.getattr("ravel_multi_index")?.call(
                (coords.clone(), dims.clone()),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("order", "F")?;
                    kwargs
                }),
            )?;
            assert_array_matches_numpy(actual_ravel.bind(py), &expected_ravel)?;

            let clip_dims = PyTuple::new(py, [4_usize, 4_usize])?;
            let clip_mode = PyTuple::new(py, ["clip", "wrap"])?;
            let actual_clip = ravel_multi_index(
                py,
                coords.clone().unbind(),
                clip_dims.clone().into_any().unbind(),
                Some(clip_mode.clone().into_any().unbind()),
                "C",
            )?;
            let expected_clip = numpy.getattr("ravel_multi_index")?.call(
                (coords, clip_dims),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("mode", clip_mode)?;
                    kwargs
                }),
            )?;
            assert_array_matches_numpy(actual_clip.bind(py), &expected_clip)?;
            Ok(())
        });
    }

    #[test]
    fn ravel_unravel_index_match_numpy_for_empty_zero_d_and_shape_preserving_cases() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let numpy = py.import("numpy")?;
            let empty_dims = PyTuple::empty(py);
            let actual_zero_d = unravel_index(
                py,
                0_i64.into_pyobject(py)?.into_any().unbind(),
                empty_dims.clone().into_any().unbind(),
                "C",
            )?;
            let expected_zero_d =
                numpy.call_method1("unravel_index", (0_i64, empty_dims.clone()))?;
            assert_eq!(
                repr_string(actual_zero_d.bind(py)),
                repr_string(&expected_zero_d)
            );

            let empty_indices = numeric_array(py, Vec::<i64>::new(), "int64");
            let empty_shape = PyTuple::new(py, [10_usize, 3_usize, 5_usize])?;
            let actual_empty = unravel_index(
                py,
                empty_indices.clone().unbind(),
                empty_shape.clone().into_any().unbind(),
                "C",
            )?;
            let expected_empty =
                numpy.call_method1("unravel_index", (empty_indices, empty_shape))?;
            assert_index_tuple_matches_numpy(actual_empty.bind(py), &expected_empty)?;

            let uint_indices = numeric_array(py, vec![vec![1_u32, 0_u32, 1_u32, 0_u32]], "uint32");
            let one_d_shape = PyTuple::new(py, [4_usize])?;
            let actual_shape_preserved = unravel_index(
                py,
                uint_indices.clone().unbind(),
                one_d_shape.clone().into_any().unbind(),
                "C",
            )?;
            let expected_shape_preserved =
                numpy.call_method1("unravel_index", (uint_indices, one_d_shape))?;
            assert_index_tuple_matches_numpy(
                actual_shape_preserved.bind(py),
                &expected_shape_preserved,
            )?;

            let actual_empty_ravel = ravel_multi_index(
                py,
                empty_dims.clone().into_any().unbind(),
                empty_dims.clone().into_any().unbind(),
                None,
                "C",
            )?;
            let expected_empty_ravel =
                numpy.call_method1("ravel_multi_index", (empty_dims.clone(), empty_dims))?;
            assert_eq!(
                repr_string(actual_empty_ravel.bind(py)),
                repr_string(&expected_empty_ravel)
            );

            let empty_coord_tuple = PyTuple::new(
                py,
                [
                    numeric_array(py, Vec::<i64>::new(), "int64"),
                    numeric_array(py, Vec::<i64>::new(), "int64"),
                ],
            )?;
            let ravel_shape = PyTuple::new(py, [5_usize, 3_usize])?;
            let actual_ravel_empty = ravel_multi_index(
                py,
                empty_coord_tuple.clone().into_any().unbind(),
                ravel_shape.clone().into_any().unbind(),
                None,
                "C",
            )?;
            let expected_ravel_empty =
                numpy.call_method1("ravel_multi_index", (empty_coord_tuple, ravel_shape))?;
            assert_array_matches_numpy(actual_ravel_empty.bind(py), &expected_ravel_empty)?;
            Ok(())
        });
    }

    #[test]
    fn ravel_unravel_index_preserve_large_exact_integer_values() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let numpy = py.import("numpy")?;
            let large = (1_i64 << 53) + 7;
            let coords = PyTuple::new(py, [0_i64, large])?;
            let dims = PyTuple::new(py, [1_usize, (large as usize) + 1])?;

            let actual_ravel = ravel_multi_index(
                py,
                coords.clone().into_any().unbind(),
                dims.clone().into_any().unbind(),
                None,
                "C",
            )?;
            let expected_ravel =
                numpy.call_method1("ravel_multi_index", (coords.clone(), dims.clone()))?;
            assert_eq!(
                repr_string(actual_ravel.bind(py)),
                repr_string(&expected_ravel)
            );

            let actual_unravel = unravel_index(
                py,
                actual_ravel.clone_ref(py),
                dims.into_any().unbind(),
                "C",
            )?;
            let expected_unravel = numpy.call_method1(
                "unravel_index",
                (
                    expected_ravel,
                    PyTuple::new(py, [1_usize, (large as usize) + 1])?,
                ),
            )?;
            assert_eq!(
                repr_string(actual_unravel.bind(py)),
                repr_string(&expected_unravel)
            );
            Ok(())
        });
    }

    #[test]
    fn ravel_unravel_index_reject_invalid_inputs_like_numpy() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let dims = PyTuple::new(py, [2_usize, 2_usize])?;

            let err = unravel_index(
                py,
                0.5_f64.into_pyobject(py)?.into_any().unbind(),
                dims.clone().into_any().unbind(),
                "C",
            )
            .unwrap_err();
            assert!(err.is_instance_of::<PyTypeError>(py));

            let err = unravel_index(
                py,
                PyList::new(py, [0_i64])?.into_any().unbind(),
                PyTuple::empty(py).into_any().unbind(),
                "C",
            )
            .unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));

            let err = ravel_multi_index(
                py,
                PyList::new(py, [1_i64, 0_i64, 1_i64, 0_i64])?
                    .into_any()
                    .unbind(),
                PyTuple::new(py, [4_usize])?.into_any().unbind(),
                None,
                "C",
            )
            .unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));

            let err = ravel_multi_index(
                py,
                PyTuple::new(py, [0.1_f64, 0.0_f64])?.into_any().unbind(),
                dims.clone().into_any().unbind(),
                None,
                "C",
            )
            .unwrap_err();
            assert!(err.is_instance_of::<PyTypeError>(py));

            let err = ravel_multi_index(
                py,
                PyTuple::new(py, [-3_i64, 1_i64])?.into_any().unbind(),
                dims.into_any().unbind(),
                None,
                "C",
            )
            .unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));
            Ok(())
        });
    }

    #[test]
    fn diag_indices_matches_numpy_default_and_ndim() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let actual_default = diag_indices(py, 4, 2)?;
            let actual_ndim = diag_indices(py, 2, 3)?;
            let numpy = py.import("numpy")?;
            let expected_default = numpy.call_method1("diag_indices", (4,))?;
            let expected_ndim = numpy.call_method1("diag_indices", (2, 3))?;

            assert_index_tuple_matches_numpy(actual_default.bind(py), &expected_default)?;
            assert_index_tuple_matches_numpy(actual_ndim.bind(py), &expected_ndim)?;
            Ok(())
        });
    }

    #[test]
    fn tri_matches_numpy_offsets_and_dtype() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let numpy = py.import("numpy")?;

            let actual_default = tri(py, 3, None, 0, None)?;
            let expected_default = numpy.call_method1("tri", (3,))?;
            assert_array_matches_numpy(actual_default.bind(py), &expected_default)?;

            let actual_neg = tri(py, 3, Some(5), -1, Some(numpy.getattr("int32")?.unbind()))?;
            let expected_neg = numpy.call_method(
                "tri",
                (3, 5),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("k", -1)?;
                    kwargs.set_item("dtype", numpy.getattr("int32")?)?;
                    kwargs
                }),
            )?;
            assert_array_matches_numpy(actual_neg.bind(py), &expected_neg)?;

            let actual_pos = tri(py, 4, Some(2), 1, Some(numpy.getattr("bool_")?.unbind()))?;
            let expected_pos = numpy.call_method(
                "tri",
                (4, 2),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("k", 1)?;
                    kwargs.set_item("dtype", numpy.getattr("bool_")?)?;
                    kwargs
                }),
            )?;
            assert_array_matches_numpy(actual_pos.bind(py), &expected_pos)
        });
    }

    #[test]
    fn rfftfreq_matches_numpy_device_and_zero_division() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let numpy = py.import("numpy")?;
            let cpu_device = "cpu".into_pyobject(py)?.into_any().unbind();
            let cuda_device = "cuda".into_pyobject(py)?.into_any().unbind();

            let actual_default = rfftfreq(py, 8, 1.0, None)?;
            let expected_default = numpy.getattr("fft")?.call_method1("rfftfreq", (8,))?;
            assert_array_matches_numpy(actual_default.bind(py), &expected_default)?;

            let actual_none = rfftfreq(py, 8, 2.0, Some(py.None()))?;
            let expected_none = numpy.getattr("fft")?.call_method(
                "rfftfreq",
                (8,),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("d", 2.0)?;
                    kwargs.set_item("device", py.None())?;
                    kwargs
                }),
            )?;
            let expected_cpu = numpy.getattr("fft")?.call_method(
                "rfftfreq",
                (8,),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("d", 2.0)?;
                    kwargs.set_item("device", "cpu")?;
                    kwargs
                }),
            )?;
            let actual_cpu_kw = rfftfreq(py, 8, 2.0, Some(cpu_device))?;
            assert_array_matches_numpy(actual_none.bind(py), &expected_none)?;
            assert_array_matches_numpy(actual_cpu_kw.bind(py), &expected_cpu)?;

            let err = rfftfreq(py, 8, 1.0, Some(cuda_device)).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));
            assert_eq!(
                err.value(py).str()?.extract::<String>()?,
                "Device not understood. Only \"cpu\" is allowed, but received: cuda"
            );

            let zero_n_err = rfftfreq(py, 0, 1.0, None).unwrap_err();
            assert!(zero_n_err.is_instance_of::<PyZeroDivisionError>(py));
            assert_eq!(
                zero_n_err.value(py).str()?.extract::<String>()?,
                "float division by zero"
            );

            let zero_d_err = rfftfreq(py, 8, 0.0, None).unwrap_err();
            assert!(zero_d_err.is_instance_of::<PyZeroDivisionError>(py));
            assert_eq!(
                zero_d_err.value(py).str()?.extract::<String>()?,
                "float division by zero"
            );

            Ok(())
        });
    }

    #[test]
    fn rfft_matches_numpy_norm_variants() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let numpy = py.import("numpy")?;
            let input = numpy.call_method1("array", (vec![1.0_f64, 2.0, 3.0, 4.0],))?;

            let actual_default = crate::rfft(py, input.clone().unbind(), None, None)?;
            let expected_default = numpy
                .getattr("fft")?
                .call_method1("rfft", (input.clone(),))?;
            assert_array_matches_numpy(actual_default.bind(py), &expected_default)?;

            let actual_backward = crate::rfft(
                py,
                input.clone().unbind(),
                None,
                Some("backward".to_string()),
            )?;
            let expected_backward = numpy.getattr("fft")?.call_method(
                "rfft",
                (input.clone(),),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("norm", "backward")?;
                    kwargs
                }),
            )?;
            assert_array_matches_numpy(actual_backward.bind(py), &expected_backward)?;

            let actual_ortho =
                crate::rfft(py, input.clone().unbind(), None, Some("ortho".to_string()))?;
            let expected_ortho = numpy.getattr("fft")?.call_method(
                "rfft",
                (input.clone(),),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("norm", "ortho")?;
                    kwargs
                }),
            )?;
            assert_array_matches_numpy(actual_ortho.bind(py), &expected_ortho)?;

            let actual_forward = crate::rfft(
                py,
                input.clone().unbind(),
                None,
                Some("forward".to_string()),
            )?;
            let expected_forward = numpy.getattr("fft")?.call_method(
                "rfft",
                (input.clone(),),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("norm", "forward")?;
                    kwargs
                }),
            )?;
            assert_array_matches_numpy(actual_forward.bind(py), &expected_forward)?;

            let err = crate::rfft(py, input.unbind(), None, Some("bad".to_string())).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));
            assert_eq!(
                err.value(py).str()?.extract::<String>()?,
                "Invalid norm value bad; should be \"backward\",\"ortho\" or \"forward\"."
            );

            Ok(())
        });
    }

    #[test]
    fn irfft_matches_numpy_norm_variants() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let numpy = py.import("numpy")?;
            let input = numpy.call_method1("array", (vec![1.0_f64, 2.0, 3.0, 4.0],))?;
            let spectrum = numpy
                .getattr("fft")?
                .call_method1("rfft", (input.clone(),))?;

            let actual_default = crate::irfft(py, spectrum.clone().unbind(), None, None)?;
            let expected_default = numpy
                .getattr("fft")?
                .call_method1("irfft", (spectrum.clone(),))?;
            assert_array_matches_numpy(actual_default.bind(py), &expected_default)?;

            let actual_backward = crate::irfft(
                py,
                spectrum.clone().unbind(),
                None,
                Some("backward".to_string()),
            )?;
            let expected_backward = numpy.getattr("fft")?.call_method(
                "irfft",
                (spectrum.clone(),),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("norm", "backward")?;
                    kwargs
                }),
            )?;
            assert_array_matches_numpy(actual_backward.bind(py), &expected_backward)?;

            let actual_ortho = crate::irfft(
                py,
                spectrum.clone().unbind(),
                None,
                Some("ortho".to_string()),
            )?;
            let expected_ortho = numpy.getattr("fft")?.call_method(
                "irfft",
                (spectrum.clone(),),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("norm", "ortho")?;
                    kwargs
                }),
            )?;
            assert_array_matches_numpy(actual_ortho.bind(py), &expected_ortho)?;

            let actual_forward = crate::irfft(
                py,
                spectrum.clone().unbind(),
                None,
                Some("forward".to_string()),
            )?;
            let expected_forward = numpy.getattr("fft")?.call_method(
                "irfft",
                (spectrum.clone(),),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("norm", "forward")?;
                    kwargs
                }),
            )?;
            assert_array_matches_numpy(actual_forward.bind(py), &expected_forward)?;

            let actual_n = crate::irfft(
                py,
                spectrum.clone().unbind(),
                Some(6),
                Some("forward".to_string()),
            )?;
            let expected_n = numpy.getattr("fft")?.call_method(
                "irfft",
                (spectrum.clone(),),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("n", 6)?;
                    kwargs.set_item("norm", "forward")?;
                    kwargs
                }),
            )?;
            assert_array_matches_numpy(actual_n.bind(py), &expected_n)?;

            let err =
                crate::irfft(py, spectrum.unbind(), None, Some("bad".to_string())).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));
            assert_eq!(
                err.value(py).str()?.extract::<String>()?,
                "Invalid norm value bad; should be \"backward\",\"ortho\" or \"forward\"."
            );

            Ok(())
        });
    }

    #[test]
    fn diag_indices_from_matches_numpy_for_object_arrays() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let arr = object_array(py, vec![vec!["a", "b"], vec!["c", "d"]]);
            let actual = diag_indices_from(py, arr.clone().unbind())?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("diag_indices_from", (arr,))?;

            assert_index_tuple_matches_numpy(actual.bind(py), &expected)?;
            Ok(())
        });
    }

    #[test]
    fn diag_indices_from_rejects_small_and_non_square_inputs() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let vector = numeric_array(py, vec![1_i64, 2_i64, 3_i64], "int64");
            let err = diag_indices_from(py, vector.unbind()).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err.to_string().contains("at least 2-d"));

            let rect = numeric_array(
                py,
                vec![vec![1_i64, 2_i64, 3_i64], vec![4_i64, 5_i64, 6_i64]],
                "int64",
            );
            let err = diag_indices_from(py, rect.unbind()).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err.to_string().contains("equal length"));
            Ok(())
        });
    }

    #[test]
    fn tril_indices_matches_numpy_with_offsets() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let actual = tril_indices(py, 4, 2, Some(5))?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("tril_indices", (4, 2, 5))?;

            assert_index_tuple_matches_numpy(actual.bind(py), &expected)?;
            Ok(())
        });
    }

    #[test]
    fn triu_indices_matches_numpy_with_offsets() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let actual = triu_indices(py, 4, 2, Some(5))?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("triu_indices", (4, 2, 5))?;

            assert_index_tuple_matches_numpy(actual.bind(py), &expected)?;
            Ok(())
        });
    }

    #[test]
    fn tril_indices_from_matches_numpy_and_rejects_non_2d() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let arr = numeric_array(
                py,
                vec![vec![1_i64, 2_i64, 3_i64], vec![4_i64, 5_i64, 6_i64]],
                "int64",
            );
            let actual = tril_indices_from(py, arr.clone().unbind(), 1)?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("tril_indices_from", (arr, 1))?;
            assert_index_tuple_matches_numpy(actual.bind(py), &expected)?;

            let cube = numeric_array(
                py,
                vec![vec![vec![1_i64, 2_i64], vec![3_i64, 4_i64]]],
                "int64",
            );
            let err = tril_indices_from(py, cube.unbind(), 0).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err.to_string().contains("2-d"));
            Ok(())
        });
    }

    #[test]
    fn triu_indices_from_matches_numpy_and_rejects_non_2d() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let arr = numeric_array(
                py,
                vec![vec![1_i64, 2_i64, 3_i64], vec![4_i64, 5_i64, 6_i64]],
                "int64",
            );
            let actual = triu_indices_from(py, arr.clone().unbind(), 1)?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("triu_indices_from", (arr, 1))?;
            assert_index_tuple_matches_numpy(actual.bind(py), &expected)?;

            let vector = numeric_array(py, vec![1_i64, 2_i64, 3_i64], "int64");
            let err = triu_indices_from(py, vector.unbind(), 0).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err.to_string().contains("2-d"));
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

    #[test]
    fn take_along_axis_matches_numpy_along_axis() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let arr = numeric_array(
                py,
                vec![vec![10.0, 30.0, 20.0], vec![60.0, 40.0, 50.0]],
                "float64",
            );
            let indices = numeric_array(py, vec![vec![0_i64, 2_i64], vec![1_i64, 0_i64]], "int64");

            let actual =
                take_along_axis(py, arr.clone().unbind(), indices.clone().unbind(), Some(1))?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("take_along_axis", (arr, indices, 1))?;

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
    fn put_along_axis_matches_numpy_scalar_values_in_place() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let arr = numeric_array(
                py,
                vec![vec![10_i64, 20_i64, 30_i64], vec![40_i64, 50_i64, 60_i64]],
                "int64",
            );
            let expected = numeric_array(
                py,
                vec![vec![10_i64, 20_i64, 30_i64], vec![40_i64, 50_i64, 60_i64]],
                "int64",
            );
            let indices = numeric_array(py, vec![vec![1_i64], vec![2_i64]], "int64");
            let scalar = 99_i64.into_pyobject(py)?.unbind();

            let actual = put_along_axis(
                py,
                arr.clone().unbind(),
                indices.clone().unbind(),
                scalar.clone_ref(py).into(),
                Some(1),
            )?;
            assert!(actual.bind(py).is_none());

            let numpy = py.import("numpy")?;
            numpy.call_method1(
                "put_along_axis",
                (expected.clone(), indices, scalar.bind(py), 1),
            )?;

            assert_eq!(
                repr_string(&arr.call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn put_along_axis_axis_none_matches_numpy_in_place() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let arr = numeric_array(
                py,
                vec![vec![10.0, 20.0, 30.0], vec![40.0, 50.0, 60.0]],
                "float64",
            );
            let expected = numeric_array(
                py,
                vec![vec![10.0, 20.0, 30.0], vec![40.0, 50.0, 60.0]],
                "float64",
            );
            let indices = numeric_array(py, vec![5_i64, 1_i64, 0_i64], "int64");
            let values = numeric_array(py, vec![7.0, 8.0, 9.0], "float64");

            let actual = put_along_axis(
                py,
                arr.clone().unbind(),
                indices.clone().unbind(),
                values.clone().unbind(),
                None,
            )?;
            assert!(actual.bind(py).is_none());

            let numpy = py.import("numpy")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("axis", py.None())?;
            numpy.call_method(
                "put_along_axis",
                (expected.clone(), indices, values),
                Some(&kwargs),
            )?;

            assert_eq!(
                repr_string(&arr.call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn put_along_axis_preserves_large_uint64_values_in_place() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let large = (1_u64 << 63) + 16387;
            let arr = numeric_array(
                py,
                vec![vec![1_u64, 2_u64, 3_u64], vec![4_u64, 5_u64, 6_u64]],
                "uint64",
            );
            let expected = numeric_array(
                py,
                vec![vec![1_u64, 2_u64, 3_u64], vec![4_u64, 5_u64, 6_u64]],
                "uint64",
            );
            let indices = numeric_array(py, vec![vec![1_i64], vec![2_i64]], "int64");
            let values = numeric_array(py, vec![vec![large], vec![large - 1]], "uint64");

            let actual = put_along_axis(
                py,
                arr.clone().unbind(),
                indices.clone().unbind(),
                values.clone().unbind(),
                Some(1),
            )?;
            assert!(actual.bind(py).is_none());

            let numpy = py.import("numpy")?;
            numpy.call_method1("put_along_axis", (expected.clone(), indices, values, 1))?;

            assert_eq!(
                arr.getattr("dtype")?.str()?.extract::<String>()?,
                expected.getattr("dtype")?.str()?.extract::<String>()?
            );
            assert_eq!(
                repr_string(&arr.call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn take_along_axis_preserves_large_uint64_values() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let large = (1_u64 << 63) + 4099;
            let arr = numeric_array(
                py,
                vec![vec![large, 3_u64, 9_u64], vec![7_u64, large - 1, 11_u64]],
                "uint64",
            );
            let indices = numeric_array(py, vec![vec![0_i64, 2_i64], vec![1_i64, 0_i64]], "int64");

            let actual =
                take_along_axis(py, arr.clone().unbind(), indices.clone().unbind(), Some(1))?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("take_along_axis", (arr, indices, 1))?;

            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn take_along_axis_axis_none_matches_numpy() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let arr = numeric_array(
                py,
                vec![vec![10.0, 30.0, 20.0], vec![60.0, 40.0, 50.0]],
                "float64",
            );
            let indices = numeric_array(py, vec![5_i64, 2_i64, 0_i64], "int64");

            let actual = take_along_axis(py, arr.clone().unbind(), indices.clone().unbind(), None)?;
            let numpy = py.import("numpy")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("axis", py.None())?;
            let expected = numpy.call_method("take_along_axis", (arr, indices), Some(&kwargs))?;

            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn compress_matches_numpy_with_short_flat_condition() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let condition = numeric_array(py, vec![0_i64, 1_i64], "int64");
            let arr = numeric_array(py, vec![10.0, 20.0, 30.0], "float64");

            let actual = compress(py, condition.clone().unbind(), arr.clone().unbind(), None)?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("compress", (condition, arr))?;

            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn compress_matches_numpy_with_short_axis_condition() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let condition = numeric_array(py, vec![0_i64, 1_i64], "int64");
            let arr = numeric_array(
                py,
                vec![vec![10.0, 20.0], vec![30.0, 40.0], vec![50.0, 60.0]],
                "float64",
            );

            let actual = compress(
                py,
                condition.clone().unbind(),
                arr.clone().unbind(),
                Some(0),
            )?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("compress", (condition, arr, 0))?;

            assert_eq!(
                actual.bind(py).getattr("shape")?.extract::<Vec<usize>>()?,
                vec![1, 2]
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn compress_preserves_large_uint64_values() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let large = (1_u64 << 63) + 8195;
            let condition = numeric_array(py, vec![1_i64, 0_i64, 1_i64, 0_i64], "int64");
            let arr = numeric_array(py, vec![large, 3_u64, large - 1, 7_u64], "uint64");

            let actual = compress(py, condition.clone().unbind(), arr.clone().unbind(), None)?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("compress", (condition, arr))?;

            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn extract_matches_numpy_boolean_mask() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let condition = numeric_array(
                py,
                vec![vec![0_i64, 1_i64], vec![1_i64, 0_i64], vec![0_i64, 1_i64]],
                "int64",
            );
            let arr = numeric_array(
                py,
                vec![vec![10.0, 20.0], vec![30.0, 40.0], vec![50.0, 60.0]],
                "float64",
            );

            let actual = extract(py, condition.clone().unbind(), arr.clone().unbind())?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("extract", (condition, arr))?;

            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn choose_matches_numpy_raise_mode() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let a = numeric_array(py, vec![0_i64, 1_i64, 0_i64, 1_i64], "int64");
            let choices = PyTuple::new(
                py,
                [
                    numeric_array(py, vec![10.0, 20.0, 30.0, 40.0], "float64")
                        .into_any()
                        .unbind(),
                    numeric_array(py, vec![1.5, 2.5, 3.5, 4.5], "float64")
                        .into_any()
                        .unbind(),
                ]
                .iter()
                .map(|item| item.bind(py)),
            )?;

            let actual = choose(
                py,
                a.clone().unbind(),
                choices.clone().into_any().unbind(),
                "raise",
            )?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("choose", (a, choices))?;

            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn choose_preserves_large_uint64_values() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let large = (1_u64 << 63) + 123;
            let a = numeric_array(py, vec![0_i64, 1_i64, 1_i64, 0_i64], "int64");
            let choices = PyTuple::new(
                py,
                [
                    numeric_array(py, vec![large, 3_u64, 5_u64, large - 1], "uint64")
                        .into_any()
                        .unbind(),
                    numeric_array(py, vec![7_u64, large - 2, large - 3, 9_u64], "uint64")
                        .into_any()
                        .unbind(),
                ]
                .iter()
                .map(|item| item.bind(py)),
            )?;

            let actual = choose(
                py,
                a.clone().unbind(),
                choices.clone().into_any().unbind(),
                "raise",
            )?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("choose", (a, choices))?;

            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn choose_matches_numpy_wrap_mode() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let a = numeric_array(py, vec![-1_i64, 0_i64, 3_i64, 4_i64], "int64");
            let choices = PyTuple::new(
                py,
                [
                    numeric_array(py, vec![10.0, 20.0, 30.0, 40.0], "float64")
                        .into_any()
                        .unbind(),
                    numeric_array(py, vec![50.0, 60.0, 70.0, 80.0], "float64")
                        .into_any()
                        .unbind(),
                    numeric_array(py, vec![90.0, 91.0, 92.0, 93.0], "float64")
                        .into_any()
                        .unbind(),
                ]
                .iter()
                .map(|item| item.bind(py)),
            )?;

            let actual = choose(
                py,
                a.clone().unbind(),
                choices.clone().into_any().unbind(),
                "wrap",
            )?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method(
                "choose",
                (a, choices),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("mode", "wrap")?;
                    kwargs
                }),
            )?;

            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn choose_matches_numpy_clip_mode() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let a = numeric_array(py, vec![-4_i64, 0_i64, 2_i64, 9_i64], "int64");
            let choices = PyTuple::new(
                py,
                [
                    numeric_array(py, vec![10.0, 20.0, 30.0, 40.0], "float64")
                        .into_any()
                        .unbind(),
                    numeric_array(py, vec![50.0, 60.0, 70.0, 80.0], "float64")
                        .into_any()
                        .unbind(),
                    numeric_array(py, vec![90.0, 91.0, 92.0, 93.0], "float64")
                        .into_any()
                        .unbind(),
                ]
                .iter()
                .map(|item| item.bind(py)),
            )?;

            let actual = choose(
                py,
                a.clone().unbind(),
                choices.clone().into_any().unbind(),
                "clip",
            )?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method(
                "choose",
                (a, choices),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("mode", "clip")?;
                    kwargs
                }),
            )?;

            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn choose_rejects_unknown_mode() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let a = numeric_array(py, vec![0_i64, 1_i64], "int64");
            let choices = PyTuple::new(
                py,
                [numeric_array(py, vec![10.0, 20.0], "float64")
                    .into_any()
                    .unbind()]
                .iter()
                .map(|item| item.bind(py)),
            )?;

            let err = choose(py, a.unbind(), choices.into_any().unbind(), "invalid").unwrap_err();
            assert!(
                err.to_string().contains("unsupported mode"),
                "unexpected error: {err}"
            );
            Ok(())
        });
    }

    #[test]
    fn select_matches_numpy_first_match_with_python_int_default() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let condlist = PyTuple::new(
                py,
                [
                    numeric_array(py, vec![true, false, false, false], "bool")
                        .into_any()
                        .unbind(),
                    numeric_array(py, vec![false, true, true, false], "bool")
                        .into_any()
                        .unbind(),
                ]
                .iter()
                .map(|item| item.bind(py)),
            )?;
            let choicelist = PyTuple::new(
                py,
                [
                    numeric_array(py, vec![10_i64, 20_i64, 30_i64, 40_i64], "int64")
                        .into_any()
                        .unbind(),
                    numeric_array(py, vec![100_i64, 200_i64, 300_i64, 400_i64], "int64")
                        .into_any()
                        .unbind(),
                ]
                .iter()
                .map(|item| item.bind(py)),
            )?;
            let default: pyo3::Py<pyo3::PyAny> = (-1_i64).into_pyobject(py)?.into_any().unbind();

            let actual = select(
                py,
                condlist.clone().into_any().unbind(),
                choicelist.clone().into_any().unbind(),
                Some(default.clone_ref(py)),
            )?;
            let numpy = py.import("numpy")?;
            let expected =
                numpy.call_method1("select", (condlist, choicelist, default.clone_ref(py)))?;

            assert_eq!(
                actual
                    .bind(py)
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?,
                expected.getattr("dtype")?.str()?.extract::<String>()?
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn select_broadcasts_conditions_and_choices_like_numpy() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let condlist = PyTuple::new(
                py,
                [
                    numeric_array(py, vec![true, false], "bool")
                        .into_any()
                        .unbind(),
                    numeric_array(py, vec![vec![false], vec![true]], "bool")
                        .into_any()
                        .unbind(),
                ]
                .iter()
                .map(|item| item.bind(py)),
            )?;
            let choicelist = PyTuple::new(
                py,
                [
                    numeric_array(py, vec![1.5, 2.5], "float64")
                        .into_any()
                        .unbind(),
                    numeric_array(py, vec![vec![10.0], vec![20.0]], "float64")
                        .into_any()
                        .unbind(),
                ]
                .iter()
                .map(|item| item.bind(py)),
            )?;

            let actual = select(
                py,
                condlist.clone().into_any().unbind(),
                choicelist.clone().into_any().unbind(),
                None,
            )?;
            let numpy = py.import("numpy")?;
            let expected = numpy.call_method1("select", (condlist, choicelist))?;

            assert_eq!(
                actual.bind(py).getattr("shape")?.extract::<Vec<usize>>()?,
                expected.getattr("shape")?.extract::<Vec<usize>>()?
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn select_preserves_large_uint64_values_with_uint64_default() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let large = (1_u64 << 63) + 123;
            let condlist = PyTuple::new(
                py,
                [
                    numeric_array(py, vec![true, false, false], "bool")
                        .into_any()
                        .unbind(),
                    numeric_array(py, vec![false, true, false], "bool")
                        .into_any()
                        .unbind(),
                ]
                .iter()
                .map(|item| item.bind(py)),
            )?;
            let choicelist = PyTuple::new(
                py,
                [
                    numeric_array(py, vec![large, 3_u64, 5_u64], "uint64")
                        .into_any()
                        .unbind(),
                    numeric_array(py, vec![7_u64, large - 1, 9_u64], "uint64")
                        .into_any()
                        .unbind(),
                ]
                .iter()
                .map(|item| item.bind(py)),
            )?;
            let numpy = py.import("numpy")?;
            let default = numpy.getattr("uint64")?.call1((large - 2,))?.unbind();

            let actual = select(
                py,
                condlist.clone().into_any().unbind(),
                choicelist.clone().into_any().unbind(),
                Some(default.clone_ref(py)),
            )?;
            let expected =
                numpy.call_method1("select", (condlist, choicelist, default.clone_ref(py)))?;

            assert_eq!(
                actual
                    .bind(py)
                    .getattr("dtype")?
                    .str()?
                    .extract::<String>()?,
                expected.getattr("dtype")?.str()?.extract::<String>()?
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn select_rejects_mismatched_condlist_and_choicelist_lengths() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let condlist = PyTuple::new(
                py,
                [numeric_array(py, vec![true, false], "bool")
                    .into_any()
                    .unbind()]
                .iter()
                .map(|item| item.bind(py)),
            )?;
            let choicelist = PyTuple::new(
                py,
                [
                    numeric_array(py, vec![1_i64, 2_i64], "int64")
                        .into_any()
                        .unbind(),
                    numeric_array(py, vec![3_i64, 4_i64], "int64")
                        .into_any()
                        .unbind(),
                ]
                .iter()
                .map(|item| item.bind(py)),
            )?;

            let err = select(
                py,
                condlist.into_any().unbind(),
                choicelist.into_any().unbind(),
                None,
            )
            .unwrap_err();
            assert!(
                err.to_string().contains("same length"),
                "unexpected error: {err}"
            );
            Ok(())
        });
    }

    #[test]
    fn fftshift_and_ifftshift_match_numpy_across_axes_odd_even_and_nd() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let fftshift_fn = module.getattr("fftshift")?;
            let ifftshift_fn = module.getattr("ifftshift")?;
            let numpy = py.import("numpy")?;
            let numpy_fftshift = numpy.getattr("fft")?.getattr("fftshift")?;
            let numpy_ifftshift = numpy.getattr("fft")?.getattr("ifftshift")?;
            let allclose = numpy.getattr("allclose")?;
            let array_equal = numpy.getattr("array_equal")?;

            // Even-length 1-D.
            let even = numpy.getattr("arange")?.call1((8_i64,))?;
            assert!(
                array_equal
                    .call1((
                        &fftshift_fn.call1((even.clone(),))?,
                        &numpy_fftshift.call1((even.clone(),))?,
                    ))?
                    .extract::<bool>()?,
                "fftshift 1-D even diverged"
            );
            assert!(
                array_equal
                    .call1((
                        &ifftshift_fn.call1((even.clone(),))?,
                        &numpy_ifftshift.call1((even.clone(),))?,
                    ))?
                    .extract::<bool>()?,
                "ifftshift 1-D even diverged"
            );

            // Odd-length 1-D — fftshift and ifftshift differ by one.
            let odd = numpy.getattr("arange")?.call1((7_i64,))?;
            assert!(
                array_equal
                    .call1((
                        &fftshift_fn.call1((odd.clone(),))?,
                        &numpy_fftshift.call1((odd.clone(),))?,
                    ))?
                    .extract::<bool>()?,
                "fftshift 1-D odd diverged"
            );
            assert!(
                array_equal
                    .call1((
                        &ifftshift_fn.call1((odd.clone(),))?,
                        &numpy_ifftshift.call1((odd.clone(),))?,
                    ))?
                    .extract::<bool>()?,
                "ifftshift 1-D odd diverged"
            );

            // 2-D default (all axes).
            let grid = numpy
                .getattr("arange")?
                .call1((12_i64,))?
                .call_method1("reshape", (3_i64, 4_i64))?;
            assert!(
                array_equal
                    .call1((
                        &fftshift_fn.call1((grid.clone(),))?,
                        &numpy_fftshift.call1((grid.clone(),))?,
                    ))?
                    .extract::<bool>()?,
                "fftshift 2-D default diverged"
            );

            // 2-D explicit single-axis (axes=0).
            let ax0_kwargs = PyDict::new(py);
            ax0_kwargs.set_item("axes", 0_i64)?;
            let ax0_kwargs_n = PyDict::new(py);
            ax0_kwargs_n.set_item("axes", 0_i64)?;
            assert!(
                array_equal
                    .call1((
                        &fftshift_fn.call((grid.clone(),), Some(&ax0_kwargs))?,
                        &numpy_fftshift.call((grid.clone(),), Some(&ax0_kwargs_n))?,
                    ))?
                    .extract::<bool>()?,
                "fftshift axes=0 diverged"
            );

            // 2-D tuple of axes.
            let tup_kwargs = PyDict::new(py);
            tup_kwargs.set_item("axes", PyTuple::new(py, [0_i64, 1_i64])?)?;
            let tup_kwargs_n = PyDict::new(py);
            tup_kwargs_n.set_item("axes", PyTuple::new(py, [0_i64, 1_i64])?)?;
            assert!(
                array_equal
                    .call1((
                        &fftshift_fn.call((grid.clone(),), Some(&tup_kwargs))?,
                        &numpy_fftshift.call((grid.clone(),), Some(&tup_kwargs_n))?,
                    ))?
                    .extract::<bool>()?,
                "fftshift axes=(0,1) diverged"
            );

            // Round-trip on odd length.
            let odd_rt = numpy.getattr("arange")?.call1((5_i64,))?;
            let shifted = fftshift_fn.call1((odd_rt.clone(),))?;
            let recovered = ifftshift_fn.call1((shifted,))?;
            assert!(
                array_equal
                    .call1((&recovered, &odd_rt))?
                    .extract::<bool>()?,
                "ifftshift(fftshift(x)) != x on odd length"
            );

            // Float input.
            let floats = numpy
                .getattr("array")?
                .call1((vec![0.25_f64, 0.5, 0.75, 1.0, 1.25, 1.5],))?;
            assert!(
                allclose
                    .call1((
                        &fftshift_fn.call1((floats.clone(),))?,
                        &numpy_fftshift.call1((floats.clone(),))?,
                    ))?
                    .extract::<bool>()?,
                "fftshift float diverged"
            );

            Ok(())
        });
    }

    #[test]
    fn norm_matches_numpy_across_ord_axis_keepdims_and_batched() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let norm_fn = module.getattr("norm")?;
            let numpy = py.import("numpy")?;
            let numpy_norm = numpy.getattr("linalg")?.getattr("norm")?;
            let isclose = numpy.getattr("isclose")?;
            let allclose = numpy.getattr("allclose")?;

            // 1-D default (L2).
            let vec = numpy.getattr("array")?.call1((vec![3.0_f64, 4.0],))?;
            assert!(
                isclose
                    .call1((
                        &norm_fn.call1((vec.clone(),))?,
                        &numpy_norm.call1((vec.clone(),))?
                    ))?
                    .extract::<bool>()?,
                "norm 1-D L2 diverged"
            );

            // 1-D with ord=1 (sum of absolute values).
            let kwargs_l1 = PyDict::new(py);
            kwargs_l1.set_item("ord", 1_i64)?;
            let kwargs_l1_n = PyDict::new(py);
            kwargs_l1_n.set_item("ord", 1_i64)?;
            assert!(
                isclose
                    .call1((
                        &norm_fn.call((vec.clone(),), Some(&kwargs_l1))?,
                        &numpy_norm.call((vec.clone(),), Some(&kwargs_l1_n))?,
                    ))?
                    .extract::<bool>()?,
                "norm 1-D L1 diverged"
            );

            // 1-D with ord=np.inf (max absolute value).
            let inf = numpy.getattr("inf")?;
            let kwargs_inf = PyDict::new(py);
            kwargs_inf.set_item("ord", &inf)?;
            let kwargs_inf_n = PyDict::new(py);
            kwargs_inf_n.set_item("ord", &inf)?;
            assert!(
                isclose
                    .call1((
                        &norm_fn.call((vec.clone(),), Some(&kwargs_inf))?,
                        &numpy_norm.call((vec.clone(),), Some(&kwargs_inf_n))?,
                    ))?
                    .extract::<bool>()?,
                "norm 1-D inf diverged"
            );

            // 2-D Frobenius (default).
            let mat = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1.0_f64, 2.0, 3.0])?,
                    PyList::new(py, [4.0_f64, 5.0, 6.0])?,
                ],
            )?,))?;
            assert!(
                isclose
                    .call1((
                        &norm_fn.call1((mat.clone(),))?,
                        &numpy_norm.call1((mat.clone(),))?,
                    ))?
                    .extract::<bool>()?,
                "norm 2-D Frobenius diverged"
            );

            // 2-D with ord='fro' explicit.
            let kwargs_fro = PyDict::new(py);
            kwargs_fro.set_item("ord", "fro")?;
            let kwargs_fro_n = PyDict::new(py);
            kwargs_fro_n.set_item("ord", "fro")?;
            assert!(
                isclose
                    .call1((
                        &norm_fn.call((mat.clone(),), Some(&kwargs_fro))?,
                        &numpy_norm.call((mat.clone(),), Some(&kwargs_fro_n))?,
                    ))?
                    .extract::<bool>()?,
                "norm 2-D fro string diverged"
            );

            // 2-D with axis=0 (column-wise norms).
            let kwargs_ax0 = PyDict::new(py);
            kwargs_ax0.set_item("axis", 0_i64)?;
            let kwargs_ax0_n = PyDict::new(py);
            kwargs_ax0_n.set_item("axis", 0_i64)?;
            assert!(
                allclose
                    .call1((
                        &norm_fn.call((mat.clone(),), Some(&kwargs_ax0))?,
                        &numpy_norm.call((mat.clone(),), Some(&kwargs_ax0_n))?,
                    ))?
                    .extract::<bool>()?,
                "norm 2-D axis=0 diverged"
            );

            // 2-D with axis=1 and keepdims=True.
            let kwargs_kd = PyDict::new(py);
            kwargs_kd.set_item("axis", 1_i64)?;
            kwargs_kd.set_item("keepdims", true)?;
            let kwargs_kd_n = PyDict::new(py);
            kwargs_kd_n.set_item("axis", 1_i64)?;
            kwargs_kd_n.set_item("keepdims", true)?;
            assert!(
                allclose
                    .call1((
                        &norm_fn.call((mat.clone(),), Some(&kwargs_kd))?,
                        &numpy_norm.call((mat.clone(),), Some(&kwargs_kd_n))?,
                    ))?
                    .extract::<bool>()?,
                "norm 2-D axis=1 keepdims diverged"
            );

            // Complex 1-D.
            let builtins = py.import("builtins")?;
            let complex_vec = numpy.getattr("array")?.call(
                (PyList::new(
                    py,
                    [
                        builtins.getattr("complex")?.call1((3.0_f64, 4.0_f64))?,
                        builtins.getattr("complex")?.call1((0.0_f64, 1.0_f64))?,
                    ],
                )?,),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("dtype", "complex128")?;
                    kwargs
                }),
            )?;
            assert!(
                isclose
                    .call1((
                        &norm_fn.call1((complex_vec.clone(),))?,
                        &numpy_norm.call1((complex_vec.clone(),))?,
                    ))?
                    .extract::<bool>()?,
                "norm complex 1-D diverged"
            );

            Ok(())
        });
    }

    #[test]
    fn cond_matches_numpy_across_p_values_and_complex() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let cond_fn = module.getattr("cond")?;
            let numpy = py.import("numpy")?;
            let numpy_cond = numpy.getattr("linalg")?.getattr("cond")?;
            let isclose = numpy.getattr("isclose")?;

            // Well-conditioned 2x2 — default p (2-norm).
            let good = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1.0_f64, 2.0])?,
                    PyList::new(py, [3.0_f64, 4.0])?,
                ],
            )?,))?;
            assert!(
                isclose
                    .call1((
                        &cond_fn.call1((good.clone(),))?,
                        &numpy_cond.call1((good.clone(),))?,
                    ))?
                    .extract::<bool>()?,
                "cond default p diverged"
            );

            // p=1 (max column sum / min column sum).
            for p in [1_i64, -1_i64, 2_i64, -2_i64] {
                let kwargs = PyDict::new(py);
                kwargs.set_item("p", p)?;
                let kwargs_n = PyDict::new(py);
                kwargs_n.set_item("p", p)?;
                assert!(
                    isclose
                        .call1((
                            &cond_fn.call((good.clone(),), Some(&kwargs))?,
                            &numpy_cond.call((good.clone(),), Some(&kwargs_n))?,
                        ))?
                        .extract::<bool>()?,
                    "cond p={p} diverged"
                );
            }

            // p='fro'.
            let fro_kwargs = PyDict::new(py);
            fro_kwargs.set_item("p", "fro")?;
            let fro_kwargs_n = PyDict::new(py);
            fro_kwargs_n.set_item("p", "fro")?;
            assert!(
                isclose
                    .call1((
                        &cond_fn.call((good.clone(),), Some(&fro_kwargs))?,
                        &numpy_cond.call((good.clone(),), Some(&fro_kwargs_n))?,
                    ))?
                    .extract::<bool>()?,
                "cond p='fro' diverged"
            );

            // p=inf and p=-inf.
            let inf = numpy.getattr("inf")?;
            let inf_kw = PyDict::new(py);
            inf_kw.set_item("p", &inf)?;
            let inf_kw_n = PyDict::new(py);
            inf_kw_n.set_item("p", &inf)?;
            assert!(
                isclose
                    .call1((
                        &cond_fn.call((good.clone(),), Some(&inf_kw))?,
                        &numpy_cond.call((good.clone(),), Some(&inf_kw_n))?,
                    ))?
                    .extract::<bool>()?,
                "cond p=inf diverged"
            );

            // Complex 2x2 default.
            let builtins = py.import("builtins")?;
            let complex_a = numpy.getattr("array")?.call(
                (PyList::new(
                    py,
                    [
                        PyList::new(
                            py,
                            [
                                builtins.getattr("complex")?.call1((1.0_f64, 1.0_f64))?,
                                builtins.getattr("complex")?.call1((2.0_f64, 0.0_f64))?,
                            ],
                        )?,
                        PyList::new(
                            py,
                            [
                                builtins.getattr("complex")?.call1((0.0_f64, -1.0_f64))?,
                                builtins.getattr("complex")?.call1((1.0_f64, 2.0_f64))?,
                            ],
                        )?,
                    ],
                )?,),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("dtype", "complex128")?;
                    kwargs
                }),
            )?;
            assert!(
                isclose
                    .call1((
                        &cond_fn.call1((complex_a.clone(),))?,
                        &numpy_cond.call1((complex_a.clone(),))?,
                    ))?
                    .extract::<bool>()?,
                "cond complex 2x2 diverged"
            );

            Ok(())
        });
    }

    #[test]
    fn masked_where_matches_numpy_across_conditions_and_copy_semantics() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let masked_where_fn = module.getattr("masked_where")?;
            let numpy = py.import("numpy")?;
            let numpy_masked_where = numpy.getattr("ma")?.getattr("masked_where")?;

            // Boolean array condition, float64 data.
            let data = numpy
                .getattr("array")?
                .call1((vec![1.0_f64, 2.0, 3.0, 4.0, 5.0],))?;
            let cond = numpy
                .getattr("array")?
                .call1((vec![false, true, false, true, false],))?;
            let actual = masked_where_fn.call1((cond.clone(), data.clone()))?;
            let expected = numpy_masked_where.call1((cond.clone(), data.clone()))?;
            assert_eq!(repr_string(&actual), repr_string(&expected));

            // Truthy condition derived from comparison.
            let greater_cond = data.call_method1("__gt__", (2.5_f64,))?;
            let actual_gt = masked_where_fn.call1((greater_cond.clone(), data.clone()))?;
            let expected_gt = numpy_masked_where.call1((greater_cond.clone(), data.clone()))?;
            assert_eq!(repr_string(&actual_gt), repr_string(&expected_gt));

            // All-False condition — no elements masked.
            let all_false = numpy
                .getattr("array")?
                .call1((vec![false, false, false, false, false],))?;
            let actual_af = masked_where_fn.call1((all_false.clone(), data.clone()))?;
            let expected_af = numpy_masked_where.call1((all_false.clone(), data.clone()))?;
            assert_eq!(repr_string(&actual_af), repr_string(&expected_af));

            // All-True condition — every element masked.
            let all_true = numpy
                .getattr("array")?
                .call1((vec![true, true, true, true, true],))?;
            let actual_at = masked_where_fn.call1((all_true.clone(), data.clone()))?;
            let expected_at = numpy_masked_where.call1((all_true.clone(), data.clone()))?;
            assert_eq!(repr_string(&actual_at), repr_string(&expected_at));

            // 2-D data + 2-D condition.
            let data_2d = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1.0_f64, 2.0, 3.0])?,
                    PyList::new(py, [4.0_f64, 5.0, 6.0])?,
                ],
            )?,))?;
            let cond_2d = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [false, true, false])?,
                    PyList::new(py, [true, false, true])?,
                ],
            )?,))?;
            let actual_2d = masked_where_fn.call1((cond_2d.clone(), data_2d.clone()))?;
            let expected_2d = numpy_masked_where.call1((cond_2d.clone(), data_2d.clone()))?;
            assert_eq!(repr_string(&actual_2d), repr_string(&expected_2d));

            // Integer dtype data.
            let int_data = numpy.getattr("array")?.call(
                (vec![10_i64, 20, 30, 40],),
                Some(&{
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("dtype", "int64")?;
                    kwargs
                }),
            )?;
            let int_cond = numpy
                .getattr("array")?
                .call1((vec![true, false, true, false],))?;
            let actual_i = masked_where_fn.call1((int_cond.clone(), int_data.clone()))?;
            let expected_i = numpy_masked_where.call1((int_cond.clone(), int_data.clone()))?;
            assert_eq!(repr_string(&actual_i), repr_string(&expected_i));

            // copy=False flag forwarded.
            let copy_kwargs = PyDict::new(py);
            copy_kwargs.set_item("copy", false)?;
            let copy_kwargs_n = PyDict::new(py);
            copy_kwargs_n.set_item("copy", false)?;
            let actual_nocopy =
                masked_where_fn.call((cond.clone(), data.clone()), Some(&copy_kwargs))?;
            let expected_nocopy =
                numpy_masked_where.call((cond.clone(), data.clone()), Some(&copy_kwargs_n))?;
            assert_eq!(repr_string(&actual_nocopy), repr_string(&expected_nocopy));

            Ok(())
        });
    }

    #[test]
    fn fftfreq_matches_numpy_across_n_d_and_device_variants() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let fftfreq_fn = module.getattr("fftfreq")?;
            let numpy = py.import("numpy")?;
            let numpy_fftfreq = numpy.getattr("fft")?.getattr("fftfreq")?;
            let allclose = numpy.getattr("allclose")?;

            // Even n with default d=1.0.
            let even = fftfreq_fn.call1((8_usize,))?;
            let expected_even = numpy_fftfreq.call1((8_i64,))?;
            assert!(
                allclose.call1((&even, &expected_even))?.extract::<bool>()?,
                "fftfreq(8) diverged"
            );

            // Odd n.
            let odd = fftfreq_fn.call1((7_usize,))?;
            let expected_odd = numpy_fftfreq.call1((7_i64,))?;
            assert!(
                allclose.call1((&odd, &expected_odd))?.extract::<bool>()?,
                "fftfreq(7) diverged"
            );

            // Non-unit d — rescales all frequencies by 1/d.
            let d_kwargs = PyDict::new(py);
            d_kwargs.set_item("d", 0.1_f64)?;
            let d_kwargs_n = PyDict::new(py);
            d_kwargs_n.set_item("d", 0.1_f64)?;
            let actual_d = fftfreq_fn.call((16_usize,), Some(&d_kwargs))?;
            let expected_d = numpy_fftfreq.call((16_i64,), Some(&d_kwargs_n))?;
            assert!(
                allclose
                    .call1((&actual_d, &expected_d))?
                    .extract::<bool>()?,
                "fftfreq(16, d=0.1) diverged"
            );

            // Large d (small frequencies).
            let big_d_kwargs = PyDict::new(py);
            big_d_kwargs.set_item("d", 1000.0_f64)?;
            let big_d_kwargs_n = PyDict::new(py);
            big_d_kwargs_n.set_item("d", 1000.0_f64)?;
            let actual_big = fftfreq_fn.call((4_usize,), Some(&big_d_kwargs))?;
            let expected_big = numpy_fftfreq.call((4_i64,), Some(&big_d_kwargs_n))?;
            assert!(
                allclose
                    .call1((&actual_big, &expected_big))?
                    .extract::<bool>()?,
                "fftfreq(4, d=1000) diverged"
            );

            // n=1 edge case.
            let single = fftfreq_fn.call1((1_usize,))?;
            let expected_single = numpy_fftfreq.call1((1_i64,))?;
            assert!(
                allclose
                    .call1((&single, &expected_single))?
                    .extract::<bool>()?,
                "fftfreq(1) diverged"
            );

            // Zero n rejected.
            let err = fftfreq_fn.call1((0_usize,)).unwrap_err();
            assert!(
                err.is_instance_of::<pyo3::exceptions::PyZeroDivisionError>(py),
                "fftfreq(0) must raise ZeroDivisionError"
            );

            // Zero d rejected.
            let zero_d = PyDict::new(py);
            zero_d.set_item("d", 0.0_f64)?;
            let err_d = fftfreq_fn.call((4_usize,), Some(&zero_d)).unwrap_err();
            assert!(
                err_d.is_instance_of::<pyo3::exceptions::PyZeroDivisionError>(py),
                "fftfreq(..., d=0) must raise ZeroDivisionError"
            );

            Ok(())
        });
    }

    #[test]
    fn masked_equal_matches_numpy_across_dtypes_and_copy() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let masked_equal_fn = module.getattr("masked_equal")?;
            let numpy = py.import("numpy")?;
            let numpy_masked_equal = numpy.getattr("ma")?.getattr("masked_equal")?;

            // Integer data with integer sentinel.
            let int_data = numpy.getattr("array")?.call(
                (vec![1_i64, 2, 3, 2, 5],),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("dtype", "int64")?;
                    kw
                }),
            )?;
            let actual_i = masked_equal_fn.call1((int_data.clone(), 2_i64))?;
            let expected_i = numpy_masked_equal.call1((int_data.clone(), 2_i64))?;
            assert_eq!(repr_string(&actual_i), repr_string(&expected_i));

            // Float data with integer sentinel.
            let float_data = numpy
                .getattr("array")?
                .call1((vec![1.0_f64, 2.0, 3.0, 2.0, 5.0],))?;
            let actual_f = masked_equal_fn.call1((float_data.clone(), 2_i64))?;
            let expected_f = numpy_masked_equal.call1((float_data.clone(), 2_i64))?;
            assert_eq!(repr_string(&actual_f), repr_string(&expected_f));

            // No matches — nothing masked.
            let actual_none = masked_equal_fn.call1((int_data.clone(), 99_i64))?;
            let expected_none = numpy_masked_equal.call1((int_data.clone(), 99_i64))?;
            assert_eq!(repr_string(&actual_none), repr_string(&expected_none));

            // All matches — everything masked.
            let uniform = numpy.getattr("array")?.call(
                (vec![7_i64, 7, 7, 7],),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("dtype", "int64")?;
                    kw
                }),
            )?;
            let actual_all = masked_equal_fn.call1((uniform.clone(), 7_i64))?;
            let expected_all = numpy_masked_equal.call1((uniform.clone(), 7_i64))?;
            assert_eq!(repr_string(&actual_all), repr_string(&expected_all));

            // 2-D input.
            let data_2d = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1_i64, 0, 1])?,
                    PyList::new(py, [0_i64, 1, 0])?,
                ],
            )?,))?;
            let actual_2d = masked_equal_fn.call1((data_2d.clone(), 0_i64))?;
            let expected_2d = numpy_masked_equal.call1((data_2d.clone(), 0_i64))?;
            assert_eq!(repr_string(&actual_2d), repr_string(&expected_2d));

            // Boolean data.
            let bool_data = numpy
                .getattr("array")?
                .call1((vec![true, false, true, false],))?;
            let actual_b = masked_equal_fn.call1((bool_data.clone(), false))?;
            let expected_b = numpy_masked_equal.call1((bool_data.clone(), false))?;
            assert_eq!(repr_string(&actual_b), repr_string(&expected_b));

            // copy=False forwarded.
            let copy_kwargs = PyDict::new(py);
            copy_kwargs.set_item("copy", false)?;
            let copy_kwargs_n = PyDict::new(py);
            copy_kwargs_n.set_item("copy", false)?;
            let actual_nocopy =
                masked_equal_fn.call((int_data.clone(), 2_i64), Some(&copy_kwargs))?;
            let expected_nocopy =
                numpy_masked_equal.call((int_data.clone(), 2_i64), Some(&copy_kwargs_n))?;
            assert_eq!(repr_string(&actual_nocopy), repr_string(&expected_nocopy));

            Ok(())
        });
    }

    #[test]
    fn masked_not_equal_matches_numpy_across_dtypes_and_copy() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let masked_not_equal_fn = module.getattr("masked_not_equal")?;
            let numpy = py.import("numpy")?;
            let numpy_masked_not_equal = numpy.getattr("ma")?.getattr("masked_not_equal")?;

            // Integer data with integer sentinel.
            let int_data = numpy.getattr("array")?.call(
                (vec![1_i64, 2, 3, 2, 5],),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("dtype", "int64")?;
                    kw
                }),
            )?;
            let actual_i = masked_not_equal_fn.call1((int_data.clone(), 2_i64))?;
            let expected_i = numpy_masked_not_equal.call1((int_data.clone(), 2_i64))?;
            assert_eq!(repr_string(&actual_i), repr_string(&expected_i));

            // Float data with integer sentinel.
            let float_data = numpy
                .getattr("array")?
                .call1((vec![1.0_f64, 2.0, 3.0, 2.0, 5.0],))?;
            let actual_f = masked_not_equal_fn.call1((float_data.clone(), 2_i64))?;
            let expected_f = numpy_masked_not_equal.call1((float_data.clone(), 2_i64))?;
            assert_eq!(repr_string(&actual_f), repr_string(&expected_f));

            // No equal entries — everything masked.
            let actual_all = masked_not_equal_fn.call1((int_data.clone(), 99_i64))?;
            let expected_all = numpy_masked_not_equal.call1((int_data.clone(), 99_i64))?;
            assert_eq!(repr_string(&actual_all), repr_string(&expected_all));

            // All equal entries — nothing masked.
            let uniform = numpy.getattr("array")?.call(
                (vec![7_i64, 7, 7, 7],),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("dtype", "int64")?;
                    kw
                }),
            )?;
            let actual_none = masked_not_equal_fn.call1((uniform.clone(), 7_i64))?;
            let expected_none = numpy_masked_not_equal.call1((uniform.clone(), 7_i64))?;
            assert_eq!(repr_string(&actual_none), repr_string(&expected_none));

            // 2-D input.
            let data_2d = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1_i64, 0, 1])?,
                    PyList::new(py, [0_i64, 1, 0])?,
                ],
            )?,))?;
            let actual_2d = masked_not_equal_fn.call1((data_2d.clone(), 0_i64))?;
            let expected_2d = numpy_masked_not_equal.call1((data_2d.clone(), 0_i64))?;
            assert_eq!(repr_string(&actual_2d), repr_string(&expected_2d));

            // Boolean data.
            let bool_data = numpy
                .getattr("array")?
                .call1((vec![true, false, true, false],))?;
            let actual_b = masked_not_equal_fn.call1((bool_data.clone(), false))?;
            let expected_b = numpy_masked_not_equal.call1((bool_data.clone(), false))?;
            assert_eq!(repr_string(&actual_b), repr_string(&expected_b));

            // copy=False forwarded.
            let copy_kwargs = PyDict::new(py);
            copy_kwargs.set_item("copy", false)?;
            let copy_kwargs_n = PyDict::new(py);
            copy_kwargs_n.set_item("copy", false)?;
            let actual_nocopy =
                masked_not_equal_fn.call((int_data.clone(), 2_i64), Some(&copy_kwargs))?;
            let expected_nocopy =
                numpy_masked_not_equal.call((int_data.clone(), 2_i64), Some(&copy_kwargs_n))?;
            assert_eq!(repr_string(&actual_nocopy), repr_string(&expected_nocopy));

            Ok(())
        });
    }

    #[test]
    fn masked_greater_matches_numpy_across_dtypes_and_copy() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let masked_greater_fn = module.getattr("masked_greater")?;
            let numpy = py.import("numpy")?;
            let numpy_masked_greater = numpy.getattr("ma")?.getattr("masked_greater")?;

            // Integer data with integer threshold.
            let int_data = numpy.getattr("array")?.call(
                (vec![1_i64, 3, 2, 5, -1],),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("dtype", "int64")?;
                    kw
                }),
            )?;
            let actual_i = masked_greater_fn.call1((int_data.clone(), 2_i64))?;
            let expected_i = numpy_masked_greater.call1((int_data.clone(), 2_i64))?;
            assert_eq!(repr_string(&actual_i), repr_string(&expected_i));

            // Float data with float threshold.
            let float_data = numpy
                .getattr("array")?
                .call1((vec![1.0_f64, 2.5, -3.0, 4.25, 0.5],))?;
            let actual_f = masked_greater_fn.call1((float_data.clone(), 1.5_f64))?;
            let expected_f = numpy_masked_greater.call1((float_data.clone(), 1.5_f64))?;
            assert_eq!(repr_string(&actual_f), repr_string(&expected_f));

            // No values above the threshold.
            let actual_none = masked_greater_fn.call1((int_data.clone(), 99_i64))?;
            let expected_none = numpy_masked_greater.call1((int_data.clone(), 99_i64))?;
            assert_eq!(repr_string(&actual_none), repr_string(&expected_none));

            // All values above the threshold.
            let positive = numpy.getattr("array")?.call(
                (vec![7_i64, 8, 9, 10],),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("dtype", "int64")?;
                    kw
                }),
            )?;
            let actual_all = masked_greater_fn.call1((positive.clone(), 0_i64))?;
            let expected_all = numpy_masked_greater.call1((positive.clone(), 0_i64))?;
            assert_eq!(repr_string(&actual_all), repr_string(&expected_all));

            // 2-D input.
            let data_2d = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1_i64, 4, 0])?,
                    PyList::new(py, [3_i64, -2, 5])?,
                ],
            )?,))?;
            let actual_2d = masked_greater_fn.call1((data_2d.clone(), 2_i64))?;
            let expected_2d = numpy_masked_greater.call1((data_2d.clone(), 2_i64))?;
            assert_eq!(repr_string(&actual_2d), repr_string(&expected_2d));

            // Boolean data uses Python ordering false < true.
            let bool_data = numpy
                .getattr("array")?
                .call1((vec![true, false, true, false],))?;
            let actual_b = masked_greater_fn.call1((bool_data.clone(), false))?;
            let expected_b = numpy_masked_greater.call1((bool_data.clone(), false))?;
            assert_eq!(repr_string(&actual_b), repr_string(&expected_b));

            // copy=False forwarded.
            let copy_kwargs = PyDict::new(py);
            copy_kwargs.set_item("copy", false)?;
            let copy_kwargs_n = PyDict::new(py);
            copy_kwargs_n.set_item("copy", false)?;
            let actual_nocopy =
                masked_greater_fn.call((int_data.clone(), 2_i64), Some(&copy_kwargs))?;
            let expected_nocopy =
                numpy_masked_greater.call((int_data.clone(), 2_i64), Some(&copy_kwargs_n))?;
            assert_eq!(repr_string(&actual_nocopy), repr_string(&expected_nocopy));

            Ok(())
        });
    }

    #[test]
    fn masked_less_matches_numpy_across_dtypes_and_thresholds() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let masked_less_fn = module.getattr("masked_less")?;
            let numpy = py.import("numpy")?;
            let numpy_masked_less = numpy.getattr("ma")?.getattr("masked_less")?;

            // Integer data with integer threshold (strict less-than).
            let int_data = numpy.getattr("array")?.call(
                (vec![1_i64, 2, 3, 4, 5],),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("dtype", "int64")?;
                    kw
                }),
            )?;
            let actual_i = masked_less_fn.call1((int_data.clone(), 3_i64))?;
            let expected_i = numpy_masked_less.call1((int_data.clone(), 3_i64))?;
            assert_eq!(repr_string(&actual_i), repr_string(&expected_i));

            // Float data with float threshold.
            let float_data = numpy
                .getattr("array")?
                .call1((vec![-1.5_f64, 0.0, 1.5, 2.5, 3.5],))?;
            let actual_f = masked_less_fn.call1((float_data.clone(), 1.0_f64))?;
            let expected_f = numpy_masked_less.call1((float_data.clone(), 1.0_f64))?;
            assert_eq!(repr_string(&actual_f), repr_string(&expected_f));

            // Threshold below all values — nothing masked.
            let actual_none = masked_less_fn.call1((int_data.clone(), 0_i64))?;
            let expected_none = numpy_masked_less.call1((int_data.clone(), 0_i64))?;
            assert_eq!(repr_string(&actual_none), repr_string(&expected_none));

            // Threshold above all values — everything masked.
            let actual_all = masked_less_fn.call1((int_data.clone(), 100_i64))?;
            let expected_all = numpy_masked_less.call1((int_data.clone(), 100_i64))?;
            assert_eq!(repr_string(&actual_all), repr_string(&expected_all));

            // 2-D input.
            let data_2d = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1.0_f64, 5.0, 2.0])?,
                    PyList::new(py, [4.0_f64, 0.5, 6.0])?,
                ],
            )?,))?;
            let actual_2d = masked_less_fn.call1((data_2d.clone(), 3.0_f64))?;
            let expected_2d = numpy_masked_less.call1((data_2d.clone(), 3.0_f64))?;
            assert_eq!(repr_string(&actual_2d), repr_string(&expected_2d));

            // Negative threshold with mixed-sign data.
            let mixed = numpy
                .getattr("array")?
                .call1((vec![-5_i64, -2, 0, 2, 5],))?;
            let actual_neg = masked_less_fn.call1((mixed.clone(), -1_i64))?;
            let expected_neg = numpy_masked_less.call1((mixed.clone(), -1_i64))?;
            assert_eq!(repr_string(&actual_neg), repr_string(&expected_neg));

            // copy=False forwarded.
            let copy_kwargs = PyDict::new(py);
            copy_kwargs.set_item("copy", false)?;
            let copy_kwargs_n = PyDict::new(py);
            copy_kwargs_n.set_item("copy", false)?;
            let actual_nocopy =
                masked_less_fn.call((int_data.clone(), 3_i64), Some(&copy_kwargs))?;
            let expected_nocopy =
                numpy_masked_less.call((int_data.clone(), 3_i64), Some(&copy_kwargs_n))?;
            assert_eq!(repr_string(&actual_nocopy), repr_string(&expected_nocopy));

            Ok(())
        });
    }

    #[test]
    fn outer_matches_numpy_across_shapes_dtypes_and_out_kwarg() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let outer_fn = module.getattr("outer")?;
            let numpy = py.import("numpy")?;
            let numpy_outer = numpy.getattr("outer")?;
            let allclose = numpy.getattr("allclose")?;
            let array_equal = numpy.getattr("array_equal")?;

            // Float 1-D × float 1-D — basic case.
            let a = numpy.getattr("array")?.call1((vec![1.0_f64, 2.0, 3.0],))?;
            let b = numpy.getattr("array")?.call1((vec![4.0_f64, 5.0],))?;
            assert!(
                allclose
                    .call1((
                        &outer_fn.call1((a.clone(), b.clone()))?,
                        &numpy_outer.call1((a.clone(), b.clone()))?,
                    ))?
                    .extract::<bool>()?,
                "outer 1-D float diverged"
            );

            // Integer 1-D × integer 1-D — exact match required.
            let ia = numpy.getattr("array")?.call(
                (vec![1_i64, 2, 3, 4],),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("dtype", "int64")?;
                    kw
                }),
            )?;
            let ib = numpy.getattr("array")?.call(
                (vec![10_i64, 20],),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("dtype", "int64")?;
                    kw
                }),
            )?;
            assert!(
                array_equal
                    .call1((
                        &outer_fn.call1((ia.clone(), ib.clone()))?,
                        &numpy_outer.call1((ia.clone(), ib.clone()))?,
                    ))?
                    .extract::<bool>()?,
                "outer 1-D int diverged"
            );

            // Higher-dimensional inputs are flattened by numpy.outer.
            let a2d = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1.0_f64, 2.0])?,
                    PyList::new(py, [3.0_f64, 4.0])?,
                ],
            )?,))?;
            let b2d = numpy.getattr("array")?.call1((vec![5.0_f64, 6.0],))?;
            assert!(
                allclose
                    .call1((
                        &outer_fn.call1((a2d.clone(), b2d.clone()))?,
                        &numpy_outer.call1((a2d.clone(), b2d.clone()))?,
                    ))?
                    .extract::<bool>()?,
                "outer 2-D flattened diverged"
            );

            // Mixed dtype promotion (int + float -> float).
            let mixed_a = numpy.getattr("array")?.call1((vec![1_i64, 2, 3],))?;
            let mixed_b = numpy.getattr("array")?.call1((vec![0.5_f64, 1.5],))?;
            assert!(
                allclose
                    .call1((
                        &outer_fn.call1((mixed_a.clone(), mixed_b.clone()))?,
                        &numpy_outer.call1((mixed_a.clone(), mixed_b.clone()))?,
                    ))?
                    .extract::<bool>()?,
                "outer mixed dtype diverged"
            );

            // Complex 1-D × complex 1-D.
            let builtins = py.import("builtins")?;
            let complex_a = numpy.getattr("array")?.call(
                (PyList::new(
                    py,
                    [
                        builtins.getattr("complex")?.call1((1.0_f64, 1.0_f64))?,
                        builtins.getattr("complex")?.call1((2.0_f64, 0.0_f64))?,
                    ],
                )?,),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("dtype", "complex128")?;
                    kw
                }),
            )?;
            let complex_b = numpy.getattr("array")?.call(
                (PyList::new(
                    py,
                    [
                        builtins.getattr("complex")?.call1((0.0_f64, 1.0_f64))?,
                        builtins.getattr("complex")?.call1((-1.0_f64, 0.0_f64))?,
                    ],
                )?,),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("dtype", "complex128")?;
                    kw
                }),
            )?;
            assert!(
                allclose
                    .call1((
                        &outer_fn.call1((complex_a.clone(), complex_b.clone()))?,
                        &numpy_outer.call1((complex_a.clone(), complex_b.clone()))?,
                    ))?
                    .extract::<bool>()?,
                "outer complex diverged"
            );

            // out= kwarg writes into a pre-allocated array of correct shape.
            let out_buf = numpy
                .getattr("zeros")?
                .call1((PyTuple::new(py, [3_i64, 2_i64])?,))?;
            let out_kwargs = PyDict::new(py);
            out_kwargs.set_item("out", out_buf.clone())?;
            let returned = outer_fn.call((a.clone(), b.clone()), Some(&out_kwargs))?;
            // numpy returns the same out buffer
            let expected_out = numpy_outer.call1((a.clone(), b.clone()))?;
            assert!(
                allclose
                    .call1((&returned, &expected_out))?
                    .extract::<bool>()?,
                "outer out= diverged"
            );

            Ok(())
        });
    }

    #[test]
    fn inner_matches_numpy_across_scalars_nd_uint64_and_error_surface() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let inner_fn = module.getattr("inner")?;
            let numpy = py.import("numpy")?;
            let numpy_inner = numpy.getattr("inner")?;

            // 1-D integer inputs return a NumPy scalar, not a 0-D ndarray.
            let a_1d = numpy.getattr("array")?.call(
                (vec![1_i64, 2, 3],),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("dtype", "int64")?;
                    kw
                }),
            )?;
            let b_1d = numpy.getattr("array")?.call(
                (vec![4_i64, 5, 6],),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("dtype", "int64")?;
                    kw
                }),
            )?;
            let actual_scalar = inner_fn.call1((a_1d.clone(), b_1d.clone()))?;
            let expected_scalar = numpy_inner.call1((a_1d.clone(), b_1d.clone()))?;
            assert_eq!(repr_string(&actual_scalar), repr_string(&expected_scalar));

            // Boolean inputs also return a NumPy scalar bool for 1-D inputs.
            let bool_a = numpy.getattr("array")?.call1((vec![true, false, true],))?;
            let bool_b = numpy.getattr("array")?.call1((vec![true, true, false],))?;
            let actual_bool = inner_fn.call1((bool_a.clone(), bool_b.clone()))?;
            let expected_bool = numpy_inner.call1((bool_a.clone(), bool_b.clone()))?;
            assert_eq!(repr_string(&actual_bool), repr_string(&expected_bool));

            // 2-D x 1-D contracts over the last axis and returns a 1-D array.
            let a_2d = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1_i64, 2, 3])?,
                    PyList::new(py, [4_i64, 5, 6])?,
                ],
            )?,))?;
            let basis = numpy.getattr("array")?.call1((vec![1_i64, 1, 1],))?;
            let actual_2d = inner_fn.call1((a_2d.clone(), basis.clone()))?;
            let expected_2d = numpy_inner.call1((a_2d.clone(), basis.clone()))?;
            assert_array_matches_numpy(&actual_2d, &expected_2d)?;

            // 2-D x 2-D contracts over the last axis of both arguments.
            let b_2d = numpy.getattr("array")?.call1((PyList::new(
                py,
                [PyList::new(py, [1_i64, 2])?, PyList::new(py, [3_i64, 4])?],
            )?,))?;
            let actual_nd = inner_fn.call1((b_2d.clone(), b_2d.clone()))?;
            let expected_nd = numpy_inner.call1((b_2d.clone(), b_2d.clone()))?;
            assert_array_matches_numpy(&actual_nd, &expected_nd)?;

            // Unsigned integer overflow behavior must match NumPy exactly.
            let big_u = numpy.getattr("array")?.call(
                (vec![1_u64 << 63],),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("dtype", "uint64")?;
                    kw
                }),
            )?;
            let two_u = numpy.getattr("array")?.call(
                (vec![2_u64],),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("dtype", "uint64")?;
                    kw
                }),
            )?;
            let actual_u = inner_fn.call1((big_u.clone(), two_u.clone()))?;
            let expected_u = numpy_inner.call1((big_u.clone(), two_u.clone()))?;
            assert_eq!(repr_string(&actual_u), repr_string(&expected_u));

            // Mismatch error text should surface unchanged.
            let mismatch_a = numpy.getattr("array")?.call1((vec![1_i64, 2, 3],))?;
            let mismatch_b = numpy.getattr("array")?.call1((vec![1_i64, 2],))?;
            let actual_err = inner_fn
                .call1((mismatch_a.clone(), mismatch_b.clone()))
                .unwrap_err();
            let expected_err = numpy_inner
                .call1((mismatch_a.clone(), mismatch_b.clone()))
                .unwrap_err();
            assert_eq!(
                actual_err.get_type(py).name()?.extract::<String>()?,
                expected_err.get_type(py).name()?.extract::<String>()?
            );
            assert_eq!(
                actual_err.value(py).str()?.extract::<String>()?,
                expected_err.value(py).str()?.extract::<String>()?
            );

            Ok(())
        });
    }

    #[test]
    fn vdot_matches_numpy_across_real_complex_and_nd_inputs() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let vdot_fn = module.getattr("vdot")?;
            let numpy = py.import("numpy")?;
            let numpy_vdot = numpy.getattr("vdot")?;
            let isclose = numpy.getattr("isclose")?;

            // Real 1-D x 1-D — basic dot product (no conjugation needed).
            let a = numpy.getattr("array")?.call1((vec![1.0_f64, 2.0, 3.0],))?;
            let b = numpy.getattr("array")?.call1((vec![4.0_f64, 5.0, 6.0],))?;
            assert!(
                isclose
                    .call1((
                        &vdot_fn.call1((a.clone(), b.clone()))?,
                        &numpy_vdot.call1((a.clone(), b.clone()))?,
                    ))?
                    .extract::<bool>()?,
                "vdot real 1-D diverged"
            );

            // Integer 1-D — exact equality.
            let ia = numpy.getattr("array")?.call1((vec![1_i64, 2, 3, 4],))?;
            let ib = numpy.getattr("array")?.call1((vec![5_i64, 6, 7, 8],))?;
            assert_eq!(
                vdot_fn.call1((ia.clone(), ib.clone()))?.extract::<i64>()?,
                numpy_vdot
                    .call1((ia.clone(), ib.clone()))?
                    .extract::<i64>()?,
                "vdot int diverged"
            );

            // Complex 1-D — vdot conjugates the first argument, so result is
            // sum(conj(a) * b). This is the critical asymmetry vs np.dot.
            let builtins = py.import("builtins")?;
            let complex_a = numpy.getattr("array")?.call(
                (PyList::new(
                    py,
                    [
                        builtins.getattr("complex")?.call1((1.0_f64, 2.0_f64))?,
                        builtins.getattr("complex")?.call1((3.0_f64, 4.0_f64))?,
                    ],
                )?,),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("dtype", "complex128")?;
                    kw
                }),
            )?;
            let complex_b = numpy.getattr("array")?.call(
                (PyList::new(
                    py,
                    [
                        builtins.getattr("complex")?.call1((5.0_f64, 0.0_f64))?,
                        builtins.getattr("complex")?.call1((0.0_f64, 1.0_f64))?,
                    ],
                )?,),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("dtype", "complex128")?;
                    kw
                }),
            )?;
            assert!(
                isclose
                    .call1((
                        &vdot_fn.call1((complex_a.clone(), complex_b.clone()))?,
                        &numpy_vdot.call1((complex_a.clone(), complex_b.clone()))?,
                    ))?
                    .extract::<bool>()?,
                "vdot complex diverged (conjugation)"
            );

            // Asymmetry verification: vdot(a, b) != vdot(b, a) for complex
            // unless both are real. They should differ by complex conjugation.
            let ab = vdot_fn.call1((complex_a.clone(), complex_b.clone()))?;
            let ba = vdot_fn.call1((complex_b.clone(), complex_a.clone()))?;
            let ab_n = numpy_vdot.call1((complex_a.clone(), complex_b.clone()))?;
            let ba_n = numpy_vdot.call1((complex_b.clone(), complex_a.clone()))?;
            assert!(
                isclose.call1((&ab, &ab_n))?.extract::<bool>()?,
                "vdot(a,b) complex diverged"
            );
            assert!(
                isclose.call1((&ba, &ba_n))?.extract::<bool>()?,
                "vdot(b,a) complex diverged"
            );

            // n-D inputs are flattened by vdot.
            let m_a = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1.0_f64, 2.0])?,
                    PyList::new(py, [3.0_f64, 4.0])?,
                ],
            )?,))?;
            let m_b = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [5.0_f64, 6.0])?,
                    PyList::new(py, [7.0_f64, 8.0])?,
                ],
            )?,))?;
            assert!(
                isclose
                    .call1((
                        &vdot_fn.call1((m_a.clone(), m_b.clone()))?,
                        &numpy_vdot.call1((m_a.clone(), m_b.clone()))?,
                    ))?
                    .extract::<bool>()?,
                "vdot n-D flattened diverged"
            );

            Ok(())
        });
    }

    #[test]
    fn kron_matches_numpy_across_shapes_and_dtypes() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let kron_fn = module.getattr("kron")?;
            let numpy = py.import("numpy")?;
            let numpy_kron = numpy.getattr("kron")?;
            let allclose = numpy.getattr("allclose")?;
            let array_equal = numpy.getattr("array_equal")?;

            // 2x2 ⊗ 2x2 — textbook Kronecker.
            let a = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1.0_f64, 2.0])?,
                    PyList::new(py, [3.0_f64, 4.0])?,
                ],
            )?,))?;
            let b = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [0.0_f64, 5.0])?,
                    PyList::new(py, [6.0_f64, 7.0])?,
                ],
            )?,))?;
            let actual = kron_fn.call1((a.clone(), b.clone()))?;
            let expected = numpy_kron.call1((a.clone(), b.clone()))?;
            assert!(
                allclose.call1((&actual, &expected))?.extract::<bool>()?,
                "kron 2x2 x 2x2 diverged"
            );

            // 1-D ⊗ 1-D — produces a 1-D array.
            let v1 = numpy.getattr("array")?.call1((vec![1.0_f64, 2.0, 3.0],))?;
            let v2 = numpy.getattr("array")?.call1((vec![4.0_f64, 5.0],))?;
            assert!(
                allclose
                    .call1((
                        &kron_fn.call1((v1.clone(), v2.clone()))?,
                        &numpy_kron.call1((v1.clone(), v2.clone()))?,
                    ))?
                    .extract::<bool>()?,
                "kron 1-D x 1-D diverged"
            );

            // Integer dtype — exact equality.
            let ia = numpy.getattr("array")?.call(
                (PyList::new(
                    py,
                    [PyList::new(py, [1_i64, 0])?, PyList::new(py, [0_i64, 1])?],
                )?,),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("dtype", "int64")?;
                    kw
                }),
            )?;
            let ib = numpy.getattr("array")?.call(
                (PyList::new(
                    py,
                    [PyList::new(py, [1_i64, 2])?, PyList::new(py, [3_i64, 4])?],
                )?,),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("dtype", "int64")?;
                    kw
                }),
            )?;
            assert!(
                array_equal
                    .call1((
                        &kron_fn.call1((ia.clone(), ib.clone()))?,
                        &numpy_kron.call1((ia.clone(), ib.clone()))?,
                    ))?
                    .extract::<bool>()?,
                "kron int diverged"
            );

            // Mismatched ranks — numpy broadcasts via leading-axis padding.
            let mat = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1.0_f64, 2.0])?,
                    PyList::new(py, [3.0_f64, 4.0])?,
                ],
            )?,))?;
            let vec1d = numpy.getattr("array")?.call1((vec![5.0_f64, 6.0],))?;
            assert!(
                allclose
                    .call1((
                        &kron_fn.call1((mat.clone(), vec1d.clone()))?,
                        &numpy_kron.call1((mat.clone(), vec1d.clone()))?,
                    ))?
                    .extract::<bool>()?,
                "kron 2-D x 1-D diverged"
            );

            // Complex dtype.
            let builtins = py.import("builtins")?;
            let complex_a = numpy.getattr("array")?.call(
                (PyList::new(
                    py,
                    [
                        builtins.getattr("complex")?.call1((1.0_f64, 1.0_f64))?,
                        builtins.getattr("complex")?.call1((0.0_f64, -1.0_f64))?,
                    ],
                )?,),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("dtype", "complex128")?;
                    kw
                }),
            )?;
            let complex_b = numpy.getattr("array")?.call(
                (PyList::new(
                    py,
                    [
                        builtins.getattr("complex")?.call1((2.0_f64, 0.0_f64))?,
                        builtins.getattr("complex")?.call1((0.0_f64, 1.0_f64))?,
                    ],
                )?,),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("dtype", "complex128")?;
                    kw
                }),
            )?;
            assert!(
                allclose
                    .call1((
                        &kron_fn.call1((complex_a.clone(), complex_b.clone()))?,
                        &numpy_kron.call1((complex_a.clone(), complex_b.clone()))?,
                    ))?
                    .extract::<bool>()?,
                "kron complex diverged"
            );

            // Identity ⊗ Identity → block-diagonal larger identity.
            let i2 = numpy.getattr("eye")?.call1((2_i64,))?;
            let i3 = numpy.getattr("eye")?.call1((3_i64,))?;
            assert!(
                allclose
                    .call1((
                        &kron_fn.call1((i2.clone(), i3.clone()))?,
                        &numpy_kron.call1((i2.clone(), i3.clone()))?,
                    ))?
                    .extract::<bool>()?,
                "kron eye x eye diverged"
            );

            Ok(())
        });
    }

    #[test]
    fn masked_inside_matches_numpy_across_intervals_and_orderings() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let masked_inside_fn = module.getattr("masked_inside")?;
            let numpy = py.import("numpy")?;
            let numpy_masked_inside = numpy.getattr("ma")?.getattr("masked_inside")?;

            // Float data with v1 < v2 — values in [v1, v2] (inclusive) are masked.
            let data = numpy
                .getattr("array")?
                .call1((vec![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0],))?;
            let actual = masked_inside_fn.call1((data.clone(), 1.0_f64, 3.0_f64))?;
            let expected = numpy_masked_inside.call1((data.clone(), 1.0_f64, 3.0_f64))?;
            assert_eq!(repr_string(&actual), repr_string(&expected));

            // Reversed v1 > v2 — np.ma normalizes to [min, max].
            let actual_rev = masked_inside_fn.call1((data.clone(), 3.0_f64, 1.0_f64))?;
            let expected_rev = numpy_masked_inside.call1((data.clone(), 3.0_f64, 1.0_f64))?;
            assert_eq!(repr_string(&actual_rev), repr_string(&expected_rev));

            // Bounds outside data range — nothing masked.
            let actual_out = masked_inside_fn.call1((data.clone(), 100.0_f64, 200.0_f64))?;
            let expected_out = numpy_masked_inside.call1((data.clone(), 100.0_f64, 200.0_f64))?;
            assert_eq!(repr_string(&actual_out), repr_string(&expected_out));

            // Bounds covering everything — full mask.
            let actual_all = masked_inside_fn.call1((data.clone(), -10.0_f64, 10.0_f64))?;
            let expected_all = numpy_masked_inside.call1((data.clone(), -10.0_f64, 10.0_f64))?;
            assert_eq!(repr_string(&actual_all), repr_string(&expected_all));

            // Integer data.
            let int_data = numpy.getattr("array")?.call(
                (vec![1_i64, 5, 10, 15, 20],),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("dtype", "int64")?;
                    kw
                }),
            )?;
            let actual_i = masked_inside_fn.call1((int_data.clone(), 5_i64, 15_i64))?;
            let expected_i = numpy_masked_inside.call1((int_data.clone(), 5_i64, 15_i64))?;
            assert_eq!(repr_string(&actual_i), repr_string(&expected_i));

            // 2-D data.
            let data_2d = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [-2.0_f64, -1.0, 0.0])?,
                    PyList::new(py, [1.0_f64, 2.0, 3.0])?,
                ],
            )?,))?;
            let actual_2d = masked_inside_fn.call1((data_2d.clone(), -1.0_f64, 1.0_f64))?;
            let expected_2d = numpy_masked_inside.call1((data_2d.clone(), -1.0_f64, 1.0_f64))?;
            assert_eq!(repr_string(&actual_2d), repr_string(&expected_2d));

            // copy=False forwarded.
            let copy_kwargs = PyDict::new(py);
            copy_kwargs.set_item("copy", false)?;
            let copy_kwargs_n = PyDict::new(py);
            copy_kwargs_n.set_item("copy", false)?;
            let actual_nocopy =
                masked_inside_fn.call((data.clone(), 1.0_f64, 3.0_f64), Some(&copy_kwargs))?;
            let expected_nocopy =
                numpy_masked_inside.call((data.clone(), 1.0_f64, 3.0_f64), Some(&copy_kwargs_n))?;
            assert_eq!(repr_string(&actual_nocopy), repr_string(&expected_nocopy));

            Ok(())
        });
    }

    #[test]
    fn masked_outside_matches_numpy_across_intervals_and_orderings() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let masked_outside_fn = module.getattr("masked_outside")?;
            let numpy = py.import("numpy")?;
            let numpy_masked_outside = numpy.getattr("ma")?.getattr("masked_outside")?;

            // Float data with v1 < v2 — values outside [v1, v2] are masked.
            let data = numpy
                .getattr("array")?
                .call1((vec![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0],))?;
            let actual = masked_outside_fn.call1((data.clone(), 1.0_f64, 3.0_f64))?;
            let expected = numpy_masked_outside.call1((data.clone(), 1.0_f64, 3.0_f64))?;
            assert_eq!(repr_string(&actual), repr_string(&expected));

            // Reversed v1 > v2 — np.ma normalizes to [min, max].
            let actual_rev = masked_outside_fn.call1((data.clone(), 3.0_f64, 1.0_f64))?;
            let expected_rev = numpy_masked_outside.call1((data.clone(), 3.0_f64, 1.0_f64))?;
            assert_eq!(repr_string(&actual_rev), repr_string(&expected_rev));

            // Bounds covering all data — nothing masked.
            let actual_none = masked_outside_fn.call1((data.clone(), -10.0_f64, 10.0_f64))?;
            let expected_none = numpy_masked_outside.call1((data.clone(), -10.0_f64, 10.0_f64))?;
            assert_eq!(repr_string(&actual_none), repr_string(&expected_none));

            // Bounds outside data range — everything masked.
            let actual_all = masked_outside_fn.call1((data.clone(), 100.0_f64, 200.0_f64))?;
            let expected_all = numpy_masked_outside.call1((data.clone(), 100.0_f64, 200.0_f64))?;
            assert_eq!(repr_string(&actual_all), repr_string(&expected_all));

            // Integer dtype.
            let int_data = numpy.getattr("array")?.call(
                (vec![1_i64, 5, 10, 15, 20],),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("dtype", "int64")?;
                    kw
                }),
            )?;
            let actual_i = masked_outside_fn.call1((int_data.clone(), 5_i64, 15_i64))?;
            let expected_i = numpy_masked_outside.call1((int_data.clone(), 5_i64, 15_i64))?;
            assert_eq!(repr_string(&actual_i), repr_string(&expected_i));

            // 2-D data with bracket interval.
            let data_2d = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [-2.0_f64, -1.0, 0.0])?,
                    PyList::new(py, [1.0_f64, 2.0, 3.0])?,
                ],
            )?,))?;
            let actual_2d = masked_outside_fn.call1((data_2d.clone(), -1.0_f64, 1.0_f64))?;
            let expected_2d = numpy_masked_outside.call1((data_2d.clone(), -1.0_f64, 1.0_f64))?;
            assert_eq!(repr_string(&actual_2d), repr_string(&expected_2d));

            // copy=False forwarded.
            let copy_kwargs = PyDict::new(py);
            copy_kwargs.set_item("copy", false)?;
            let copy_kwargs_n = PyDict::new(py);
            copy_kwargs_n.set_item("copy", false)?;
            let actual_nocopy =
                masked_outside_fn.call((data.clone(), 1.0_f64, 3.0_f64), Some(&copy_kwargs))?;
            let expected_nocopy = numpy_masked_outside
                .call((data.clone(), 1.0_f64, 3.0_f64), Some(&copy_kwargs_n))?;
            assert_eq!(repr_string(&actual_nocopy), repr_string(&expected_nocopy));

            Ok(())
        });
    }

    #[test]
    fn masked_greater_equal_matches_numpy_across_dtypes_and_thresholds() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let masked_ge_fn = module.getattr("masked_greater_equal")?;
            let numpy = py.import("numpy")?;
            let numpy_masked_ge = numpy.getattr("ma")?.getattr("masked_greater_equal")?;

            // Integer data with integer threshold (>= masks).
            let int_data = numpy.getattr("array")?.call(
                (vec![1_i64, 2, 3, 4, 5],),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("dtype", "int64")?;
                    kw
                }),
            )?;
            let actual_i = masked_ge_fn.call1((int_data.clone(), 3_i64))?;
            let expected_i = numpy_masked_ge.call1((int_data.clone(), 3_i64))?;
            assert_eq!(repr_string(&actual_i), repr_string(&expected_i));

            // Float data with float threshold.
            let float_data = numpy
                .getattr("array")?
                .call1((vec![-1.5_f64, 0.0, 1.5, 2.5, 3.5],))?;
            let actual_f = masked_ge_fn.call1((float_data.clone(), 1.0_f64))?;
            let expected_f = numpy_masked_ge.call1((float_data.clone(), 1.0_f64))?;
            assert_eq!(repr_string(&actual_f), repr_string(&expected_f));

            // Threshold above all values — nothing masked.
            let actual_none = masked_ge_fn.call1((int_data.clone(), 100_i64))?;
            let expected_none = numpy_masked_ge.call1((int_data.clone(), 100_i64))?;
            assert_eq!(repr_string(&actual_none), repr_string(&expected_none));

            // Threshold below all values — everything masked.
            let actual_all = masked_ge_fn.call1((int_data.clone(), 0_i64))?;
            let expected_all = numpy_masked_ge.call1((int_data.clone(), 0_i64))?;
            assert_eq!(repr_string(&actual_all), repr_string(&expected_all));

            // Boundary: threshold equal to a present value (>= includes it).
            let actual_eq = masked_ge_fn.call1((int_data.clone(), 3_i64))?;
            let expected_eq = numpy_masked_ge.call1((int_data.clone(), 3_i64))?;
            assert_eq!(repr_string(&actual_eq), repr_string(&expected_eq));

            // 2-D input.
            let data_2d = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1.0_f64, 5.0, 2.0])?,
                    PyList::new(py, [4.0_f64, 0.5, 6.0])?,
                ],
            )?,))?;
            let actual_2d = masked_ge_fn.call1((data_2d.clone(), 3.0_f64))?;
            let expected_2d = numpy_masked_ge.call1((data_2d.clone(), 3.0_f64))?;
            assert_eq!(repr_string(&actual_2d), repr_string(&expected_2d));

            // copy=False forwarded.
            let copy_kwargs = PyDict::new(py);
            copy_kwargs.set_item("copy", false)?;
            let copy_kwargs_n = PyDict::new(py);
            copy_kwargs_n.set_item("copy", false)?;
            let actual_nocopy = masked_ge_fn.call((int_data.clone(), 3_i64), Some(&copy_kwargs))?;
            let expected_nocopy =
                numpy_masked_ge.call((int_data.clone(), 3_i64), Some(&copy_kwargs_n))?;
            assert_eq!(repr_string(&actual_nocopy), repr_string(&expected_nocopy));

            Ok(())
        });
    }

    #[test]
    fn count_masked_matches_numpy_scalars_axes_and_error_surface() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let count_masked_fn = module.getattr("count_masked")?;
            let numpy = py.import("numpy")?;
            let ma = numpy.getattr("ma")?;
            let numpy_count_masked = ma.getattr("count_masked")?;
            let ma_array = ma.getattr("array")?;

            // Default path returns a NumPy scalar count.
            let masked_1d = ma_array.call(
                (vec![1_i64, 2, 3, 4],),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("mask", vec![false, true, false, true])?;
                    kw
                }),
            )?;
            let actual_default = count_masked_fn.call1((masked_1d.clone(),))?;
            let expected_default = numpy_count_masked.call1((masked_1d.clone(),))?;
            assert_eq!(repr_string(&actual_default), repr_string(&expected_default));

            // Axis reductions return ndarrays with NumPy's integer count dtype.
            let masked_2d = ma_array.call(
                (PyList::new(
                    py,
                    [
                        PyList::new(py, [1_i64, 2, 3])?,
                        PyList::new(py, [4_i64, 5, 6])?,
                    ],
                )?,),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item(
                        "mask",
                        PyList::new(
                            py,
                            [
                                PyList::new(py, [false, true, false])?,
                                PyList::new(py, [true, false, true])?,
                            ],
                        )?,
                    )?;
                    kw
                }),
            )?;
            let actual_axis0 = count_masked_fn.call1((masked_2d.clone(), 0_i64))?;
            let expected_axis0 = numpy_count_masked.call1((masked_2d.clone(), 0_i64))?;
            assert_eq!(repr_string(&actual_axis0), repr_string(&expected_axis0));

            let actual_axis1 = count_masked_fn.call1((masked_2d.clone(), 1_i64))?;
            let expected_axis1 = numpy_count_masked.call1((masked_2d.clone(), 1_i64))?;
            assert_eq!(repr_string(&actual_axis1), repr_string(&expected_axis1));

            // Tuple axes collapse to a scalar when all dimensions reduce.
            let axes = PyTuple::new(py, [0_i64, 1_i64])?;
            let actual_tuple = count_masked_fn.call1((masked_2d.clone(), axes))?;
            let expected_tuple = numpy_count_masked.call1((masked_2d.clone(), (0_i64, 1_i64)))?;
            assert_eq!(repr_string(&actual_tuple), repr_string(&expected_tuple));

            // Plain ndarrays without a mask should report zero masked entries.
            let plain = numpy.getattr("array")?.call1((vec![10_i64, 20, 30],))?;
            let actual_plain = count_masked_fn.call1((plain.clone(),))?;
            let expected_plain = numpy_count_masked.call1((plain.clone(),))?;
            assert_eq!(repr_string(&actual_plain), repr_string(&expected_plain));

            // Invalid axis errors should surface unchanged.
            let actual_err = count_masked_fn
                .call1((masked_2d.clone(), 2_i64))
                .unwrap_err();
            let expected_err = numpy_count_masked
                .call1((masked_2d.clone(), 2_i64))
                .unwrap_err();
            assert_eq!(
                actual_err.get_type(py).name()?.extract::<String>()?,
                expected_err.get_type(py).name()?.extract::<String>()?
            );
            assert_eq!(
                actual_err.value(py).str()?.extract::<String>()?,
                expected_err.value(py).str()?.extract::<String>()?
            );

            Ok(())
        });
    }

    #[test]
    fn masked_less_equal_matches_numpy_across_dtypes_and_thresholds() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let masked_le_fn = module.getattr("masked_less_equal")?;
            let numpy = py.import("numpy")?;
            let numpy_masked_le = numpy.getattr("ma")?.getattr("masked_less_equal")?;

            // Integer data with integer threshold (<= masks).
            let int_data = numpy.getattr("array")?.call(
                (vec![1_i64, 2, 3, 4, 5],),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("dtype", "int64")?;
                    kw
                }),
            )?;
            let actual_i = masked_le_fn.call1((int_data.clone(), 3_i64))?;
            let expected_i = numpy_masked_le.call1((int_data.clone(), 3_i64))?;
            assert_eq!(repr_string(&actual_i), repr_string(&expected_i));

            // Float data with float threshold.
            let float_data = numpy
                .getattr("array")?
                .call1((vec![-1.5_f64, 0.0, 1.5, 2.5, 3.5],))?;
            let actual_f = masked_le_fn.call1((float_data.clone(), 1.0_f64))?;
            let expected_f = numpy_masked_le.call1((float_data.clone(), 1.0_f64))?;
            assert_eq!(repr_string(&actual_f), repr_string(&expected_f));

            // Threshold below all values — nothing masked.
            let actual_none = masked_le_fn.call1((int_data.clone(), 0_i64))?;
            let expected_none = numpy_masked_le.call1((int_data.clone(), 0_i64))?;
            assert_eq!(repr_string(&actual_none), repr_string(&expected_none));

            // Threshold above all values — everything masked.
            let actual_all = masked_le_fn.call1((int_data.clone(), 100_i64))?;
            let expected_all = numpy_masked_le.call1((int_data.clone(), 100_i64))?;
            assert_eq!(repr_string(&actual_all), repr_string(&expected_all));

            // Boundary: threshold equal to a present value (<= includes it).
            let actual_eq = masked_le_fn.call1((int_data.clone(), 3_i64))?;
            let expected_eq = numpy_masked_le.call1((int_data.clone(), 3_i64))?;
            assert_eq!(repr_string(&actual_eq), repr_string(&expected_eq));

            // 2-D input.
            let data_2d = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1.0_f64, 5.0, 2.0])?,
                    PyList::new(py, [4.0_f64, 0.5, 6.0])?,
                ],
            )?,))?;
            let actual_2d = masked_le_fn.call1((data_2d.clone(), 3.0_f64))?;
            let expected_2d = numpy_masked_le.call1((data_2d.clone(), 3.0_f64))?;
            assert_eq!(repr_string(&actual_2d), repr_string(&expected_2d));

            // copy=False forwarded.
            let copy_kwargs = PyDict::new(py);
            copy_kwargs.set_item("copy", false)?;
            let copy_kwargs_n = PyDict::new(py);
            copy_kwargs_n.set_item("copy", false)?;
            let actual_nocopy = masked_le_fn.call((int_data.clone(), 3_i64), Some(&copy_kwargs))?;
            let expected_nocopy =
                numpy_masked_le.call((int_data.clone(), 3_i64), Some(&copy_kwargs_n))?;
            assert_eq!(repr_string(&actual_nocopy), repr_string(&expected_nocopy));

            Ok(())
        });
    }

    #[test]
    fn masked_values_matches_numpy_across_tolerances_and_dtypes() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;
            let masked_values_fn = module.getattr("masked_values")?;
            let numpy = py.import("numpy")?;
            let numpy_masked_values = numpy.getattr("ma")?.getattr("masked_values")?;

            // Float data with float sentinel — defaults rtol=1e-5, atol=1e-8.
            let float_data = numpy
                .getattr("array")?
                .call1((vec![1.0_f64, 1.5, 2.0, 1.5000001, 3.0],))?;
            let actual = masked_values_fn.call1((float_data.clone(), 1.5_f64))?;
            let expected = numpy_masked_values.call1((float_data.clone(), 1.5_f64))?;
            assert_eq!(repr_string(&actual), repr_string(&expected));

            // Loose rtol — should mask the slightly-off value 1.5000001.
            let loose_kwargs = PyDict::new(py);
            loose_kwargs.set_item("rtol", 1e-3_f64)?;
            loose_kwargs.set_item("atol", 1e-3_f64)?;
            let loose_kwargs_n = PyDict::new(py);
            loose_kwargs_n.set_item("rtol", 1e-3_f64)?;
            loose_kwargs_n.set_item("atol", 1e-3_f64)?;
            let actual_loose = masked_values_fn.call(
                (float_data.clone(), 1.5_f64),
                Some(&loose_kwargs),
            )?;
            let expected_loose = numpy_masked_values.call(
                (float_data.clone(), 1.5_f64),
                Some(&loose_kwargs_n),
            )?;
            assert_eq!(repr_string(&actual_loose), repr_string(&expected_loose));

            // Tight tolerances — only exact matches.
            let tight_kwargs = PyDict::new(py);
            tight_kwargs.set_item("rtol", 0.0_f64)?;
            tight_kwargs.set_item("atol", 0.0_f64)?;
            let tight_kwargs_n = PyDict::new(py);
            tight_kwargs_n.set_item("rtol", 0.0_f64)?;
            tight_kwargs_n.set_item("atol", 0.0_f64)?;
            let actual_tight = masked_values_fn.call(
                (float_data.clone(), 1.5_f64),
                Some(&tight_kwargs),
            )?;
            let expected_tight = numpy_masked_values.call(
                (float_data.clone(), 1.5_f64),
                Some(&tight_kwargs_n),
            )?;
            assert_eq!(repr_string(&actual_tight), repr_string(&expected_tight));

            // Integer data — falls back to exact equality.
            let int_data = numpy.getattr("array")?.call(
                (vec![1_i64, 2, 3, 2, 5],),
                Some(&{
                    let kw = PyDict::new(py);
                    kw.set_item("dtype", "int64")?;
                    kw
                }),
            )?;
            let actual_i = masked_values_fn.call1((int_data.clone(), 2_i64))?;
            let expected_i = numpy_masked_values.call1((int_data.clone(), 2_i64))?;
            assert_eq!(repr_string(&actual_i), repr_string(&expected_i));

            // 2-D float input.
            let data_2d = numpy.getattr("array")?.call1((PyList::new(
                py,
                [
                    PyList::new(py, [1.0_f64, 2.0, 3.0])?,
                    PyList::new(py, [2.0_f64, 4.0, 5.0])?,
                ],
            )?,))?;
            let actual_2d = masked_values_fn.call1((data_2d.clone(), 2.0_f64))?;
            let expected_2d = numpy_masked_values.call1((data_2d.clone(), 2.0_f64))?;
            assert_eq!(repr_string(&actual_2d), repr_string(&expected_2d));

            // shrink=False forwarded.
            let no_match_data = numpy
                .getattr("array")?
                .call1((vec![10.0_f64, 20.0, 30.0],))?;
            let shrink_kwargs = PyDict::new(py);
            shrink_kwargs.set_item("shrink", false)?;
            let shrink_kwargs_n = PyDict::new(py);
            shrink_kwargs_n.set_item("shrink", false)?;
            let actual_shrink = masked_values_fn.call(
                (no_match_data.clone(), 99.0_f64),
                Some(&shrink_kwargs),
            )?;
            let expected_shrink = numpy_masked_values.call(
                (no_match_data.clone(), 99.0_f64),
                Some(&shrink_kwargs_n),
            )?;
            assert_eq!(
                repr_string(&actual_shrink),
                repr_string(&expected_shrink)
            );

            Ok(())
        });
    }
}
