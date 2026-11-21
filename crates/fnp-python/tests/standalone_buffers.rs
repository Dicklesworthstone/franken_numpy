//! Standalone array exports must remain live through typed item assignment.

mod common;

use common::with_fnp_and_numpy;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[test]
fn existing_buffer_and_array_interface_views_observe_mutation() {
    with_fnp_and_numpy(|py, module, numpy| {
        let locals = PyDict::new(py);
        locals.set_item("fnp", module)?;
        locals.set_item("np", numpy)?;
        py.run(
            pyo3::ffi::c_str!(
                r#"
import gc

class InterfaceOnly:
    def __init__(self, owner):
        self.owner = owner
    @property
    def __array_interface__(self):
        return self.owner.__array_interface__

for dtype in ['bool', 'int8', 'uint8', 'int16', 'uint16', 'int32',
              'uint32', 'int64', 'uint64', 'float16', 'float32', 'float64']:
    expected = np.arange(6).reshape(2, 3).astype(dtype)
    a = fnp.PyNdArray.from_bytes(expected.tobytes(), dtype=dtype, shape=(2, 3))
    before = a.__array_interface__['data'][0]
    exported = memoryview(a)
    byte_view = exported.cast('B')
    numpy_view = np.asarray(a)
    interface_view = np.asarray(InterfaceOnly(a))
    assert numpy_view.__array_interface__['data'][0] == before
    assert interface_view.__array_interface__['data'][0] == before
    assert exported.readonly and not numpy_view.flags.writeable
    for value in [17, 0, 31, 7]:
        a[1, 2] = value
        expected[1, 2] = value
        assert a.__array_interface__['data'][0] == before, dtype
        assert exported.shape == (2, 3) and exported.strides == expected.strides
        assert exported.tobytes() == expected.tobytes(), dtype
        assert byte_view.tobytes() == expected.tobytes(), dtype
        assert a.tobytes() == expected.tobytes(), dtype
        np.testing.assert_array_equal(numpy_view, expected)
        np.testing.assert_array_equal(interface_view, expected)
    independent = a.copy()
    independent[0, 0] = 1
    assert a.tobytes() == expected.tobytes()
    del a, independent
    gc.collect()
    assert exported.tobytes() == expected.tobytes(), dtype
    np.testing.assert_array_equal(numpy_view, expected)
    np.testing.assert_array_equal(interface_view, expected)
    byte_view.release()
    exported.release()
"#
            ),
            Some(&locals),
            Some(&locals),
        )?;
        Ok(())
    });
}

#[test]
fn standalone_indexing_and_assignment_preserve_wide_integers() {
    with_fnp_and_numpy(|py, module, numpy| {
        let locals = PyDict::new(py);
        locals.set_item("fnp", module)?;
        locals.set_item("np", numpy)?;
        py.run(
            pyo3::ffi::c_str!(
                r#"
cases = [
    ('int64', [2**53+1, 2**63-1, -2**63+1, -2**53-3], [-2**63, 2**63-1]),
    ('uint64', [2**64-1, 2**63+3, 2**53+5, 0], [2**64-3, 2**63+1]),
]
for dtype, values, replacements in cases:
    expected = np.array(values, dtype=dtype).reshape(2, 2)
    a = fnp.PyNdArray.from_bytes(expected.tobytes(), dtype=dtype, shape=(2, 2))
    exported = memoryview(a)
    view = np.asarray(a)
    for row in [0, 1, -1, -2]:
        assert a[row].tolist() == expected[row].tolist()
        assert a[row].tobytes() == expected[row].tobytes()
    for value in replacements:
        a[-1, 0] = value
        expected[-1, 0] = value
        assert a.tolist() == expected.tolist()
        assert a[1, 0] == int(expected[1, 0])
        assert exported.tobytes() == expected.tobytes()
        assert view.tobytes() == expected.tobytes()
    # Writing a small float must not round the other wide integer elements.
    a[0, 1] = 12.75
    expected[0, 1] = 12.75
    assert a.tolist() == expected.tolist()
    assert exported.tobytes() == expected.tobytes()
    one_dimensional = fnp.PyNdArray.from_bytes(expected[1].tobytes(), dtype=dtype)
    one_dimensional[0] = replacements[0]
    assert one_dimensional[0] == replacements[0]
    assert one_dimensional[1] == int(expected[1, 1])
"#
            ),
            Some(&locals),
            Some(&locals),
        )?;
        Ok(())
    });
}

#[test]
fn failed_standalone_assignments_leave_all_exports_unchanged() {
    with_fnp_and_numpy(|py, module, numpy| {
        let locals = PyDict::new(py);
        locals.set_item("fnp", module)?;
        locals.set_item("np", numpy)?;
        py.run(
            pyo3::ffi::c_str!(
                r#"
for dtype in ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']:
    bounds = np.iinfo(dtype)
    expected = np.array([1, 2], dtype=dtype)
    a = fnp.PyNdArray.from_bytes(expected.tobytes(), dtype=dtype)
    exported = memoryview(a)
    view = np.asarray(a)
    address = a.__array_interface__['data'][0]
    for invalid in [int(bounds.min)-1, int(bounds.max)+1, float('nan'), float('inf'), object()]:
        try:
            expected[0] = invalid
        except Exception as error:
            expected_error = type(error)
        else:
            raise AssertionError(('oracle accepted invalid scalar', dtype, invalid))
        try:
            a[0] = invalid
        except Exception as error:
            assert type(error) is expected_error, (dtype, invalid, type(error), expected_error)
        else:
            raise AssertionError(('assignment accepted invalid scalar', dtype, invalid))
        assert a.__array_interface__['data'][0] == address
        assert a.tobytes() == expected.tobytes()
        assert exported.tobytes() == expected.tobytes()
        assert view.tobytes() == expected.tobytes()

a = fnp.PyNdArray([1.0, 2.0])
view = memoryview(a)
class Reentrant:
    def __float__(self):
        memoryview(a)
        return 3.0
try:
    a[0] = Reentrant()
except RuntimeError:
    pass
else:
    raise AssertionError('reentrant buffer export must reject the active mutable borrow')
assert view.tolist() == [1.0, 2.0]
assert a.tolist() == [1.0, 2.0]
"#
            ),
            Some(&locals),
            Some(&locals),
        )?;
        Ok(())
    });
}

#[test]
fn typed_scalar_and_zero_dimensional_integer_assignment_matches_numpy() {
    with_fnp_and_numpy(|py, module, numpy| {
        let locals = PyDict::new(py);
        locals.set_item("fnp", module)?;
        locals.set_item("np", numpy)?;
        py.run(
            pyo3::ffi::c_str!(
                r#"
import warnings

scalars = [
    np.int8(-1), np.uint8(255), np.int64(-1), np.int64(-129),
    np.int64(128), np.int64(2**53+1), np.int64(-2**63), np.int64(2**63-1),
    np.uint64(2**53+1), np.uint64(2**63+1), np.uint64(2**64-1),
    np.bool_(False), np.bool_(True),
]
float_values = [
    -129.75, -128.75, -1.75, -0.75, -0.0, 0.75, 12.75,
    127.75, 128.75, 255.75, 256.75,
    -(2.0**31), 2.0**31, 2.0**32,
    -(2.0**63), 2.0**63, 2.0**64,
    float('nan'), float('inf'), float('-inf'),
]
with warnings.catch_warnings(), np.errstate(all='ignore'):
    warnings.simplefilter('ignore')
    for dtype in [np.float16, np.float32, np.float64]:
        scalars.extend(dtype(value) for value in float_values)
        for boundary in [2.0**15, 2.0**16, 2.0**31, 2.0**32, 2.0**63, 2.0**64]:
            point = dtype(boundary)
            scalars.extend([
                np.nextafter(point, dtype('-inf')),
                np.nextafter(point, dtype('inf')),
            ])

cases = []
for scalar in scalars:
    # Scalar and 0-D ndarray assignment are distinct NumPy contracts. For
    # signed destinations, np.int64(128) raises when assigned to int8 whereas
    # a zero-dimensional int64 ndarray wraps. Unsigned destinations also
    # distinguish strongly typed NumPy scalars from builtin Python values.
    label = (str(scalar.dtype), repr(scalar))
    array = np.asarray(scalar).copy()
    readonly = array.copy()
    readonly.flags.writeable = False
    cases.extend([
        (("scalar", *label), scalar),
        (("zero-dimensional", *label), array),
        (("readonly-zero-dimensional", *label), readonly),
    ])

for dtype in ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']:
    bounds = np.iinfo(dtype)
    expected = np.array([1, bounds.max, 0, bounds.min], dtype=dtype)
    a = fnp.PyNdArray.from_bytes(expected.tobytes(), dtype=dtype)
    exported = memoryview(a)
    byte_view = exported.cast('B')
    numpy_view = np.asarray(a)
    address = a.__array_interface__['data'][0]
    for label, value in cases:
        before = expected.tobytes()
        expected_error = None
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            try:
                expected[0] = value
            except Exception as error:
                expected_error = type(error)
                assert expected.tobytes() == before, (dtype, label, 'oracle rollback')
            try:
                a[0] = value
            except Exception as error:
                assert type(error) is expected_error, (dtype, label, type(error), expected_error)
            else:
                assert expected_error is None, (dtype, label, 'missing exception', expected_error)
        assert a.__array_interface__['data'][0] == address, (dtype, label, 'address')
        assert a.tobytes() == expected.tobytes(), (dtype, label, 'owned bytes')
        assert exported.tobytes() == expected.tobytes(), (dtype, label, 'exported bytes')
        assert byte_view.tobytes() == expected.tobytes(), (dtype, label, 'byte view')
        assert numpy_view.tobytes() == expected.tobytes(), (dtype, label, 'numpy view')
        assert a.tolist() == expected.tolist(), (dtype, label, 'typed values')
    byte_view.release()
    exported.release()
"#
            ),
            Some(&locals),
            Some(&locals),
        )?;
        Ok(())
    });
}

#[test]
fn standalone_float_and_boolean_scalar_conversions_match_numpy() {
    with_fnp_and_numpy(|py, module, numpy| {
        let locals = PyDict::new(py);
        locals.set_item("fnp", module)?;
        locals.set_item("np", numpy)?;
        py.run(
            pyo3::ffi::c_str!(
                r#"
import warnings

class FloatLike:
    def __float__(self):
        return 3.25

class BrokenFloat:
    def __float__(self):
        raise RuntimeError('conversion failed')

class BrokenTruth:
    def __bool__(self):
        raise ValueError('truth conversion failed')

float_cases = [
    None, False, True, -0.0, 12.75, -18.5, 2**53+1,
    np.uint64(2**64-1), np.float16(1.25), np.float32(1.25), np.float64(-2.75),
    np.array(2.75), float('nan'), float('inf'), float('-inf'),
    '1.25', 'nan', '-inf', b'2.5', FloatLike(), BrokenFloat(),
    object(), 'not-a-number', complex(1, 2),
]
bool_cases = [
    None, False, True, 0, 1, -5, -0.0, 0.75, float('nan'), float('inf'),
    '', '0', 'false', b'', b'0', complex(0, 0), complex(0, 1),
    np.bool_(False), np.int64(0), np.uint64(2**64-1), np.float32(float('nan')),
    np.array(0), np.array(3), object(), BrokenTruth(),
]

for dtype in ['float16', 'float32', 'float64', 'bool']:
    expected = np.array([1, -0.0, 3], dtype=dtype)
    a = fnp.PyNdArray.from_bytes(expected.tobytes(), dtype=dtype)
    exported = memoryview(a)
    numpy_view = np.asarray(a)
    address = a.__array_interface__['data'][0]
    for value in bool_cases if dtype == 'bool' else float_cases:
        label = (dtype, type(value).__name__, repr(value))
        before = expected.tobytes()
        expected_error = None
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            try:
                expected[0] = value
            except Exception as error:
                expected_error = type(error)
                assert expected.tobytes() == before, (label, 'oracle rollback')
            try:
                a[0] = value
            except Exception as error:
                assert type(error) is expected_error, (label, type(error), expected_error)
            else:
                assert expected_error is None, (label, 'missing exception', expected_error)
        assert a.__array_interface__['data'][0] == address, (label, 'address')
        assert a.tobytes() == expected.tobytes(), (label, 'owned bytes')
        assert exported.tobytes() == expected.tobytes(), (label, 'exported bytes')
        assert numpy_view.tobytes() == expected.tobytes(), (label, 'numpy view')
        np.testing.assert_array_equal(np.asarray(a.tolist()), expected)
    exported.release()
"#
            ),
            Some(&locals),
            Some(&locals),
        )?;
        Ok(())
    });
}
