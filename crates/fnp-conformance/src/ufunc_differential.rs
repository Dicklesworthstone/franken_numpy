#![forbid(unsafe_code)]

use fnp_dtype::DType;
use fnp_ufunc::{BinaryOp, UFuncArray, UnaryOp};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const PY_CAPTURE_SCRIPT: &str = r#"
import json
import importlib
import sys
import math
import bisect

input_path = sys.argv[1]
output_path = sys.argv[2]
legacy_root = sys.argv[3]

with open(input_path, 'r', encoding='utf-8') as fh:
    cases = json.load(fh)

oracle_source = 'legacy'
np = None

try:
    if 'numpy' in sys.modules:
        del sys.modules['numpy']
    if legacy_root not in sys.path:
        sys.path.insert(0, legacy_root)
    np = importlib.import_module('numpy')
    _ = np.arange(1)
    np_file = str(getattr(np, '__file__', ''))
    if 'legacy_numpy_code' not in np_file:
        oracle_source = 'system'
except Exception:
    try:
        oracle_source = 'system'
        if 'numpy' in sys.modules:
            del sys.modules['numpy']
        if legacy_root in sys.path:
            sys.path.remove(legacy_root)
        np = importlib.import_module('numpy')
        _ = np.arange(1)
    except Exception:
        oracle_source = 'pure_python_fallback'
        np = None

def py_broadcast_shape(lhs, rhs):
    nd = max(len(lhs), len(rhs))
    out = []
    for i in range(nd):
        l = lhs[-1 - i] if i < len(lhs) else 1
        r = rhs[-1 - i] if i < len(rhs) else 1
        if l == r:
            out.append(l)
        elif l == 1:
            out.append(r)
        elif r == 1:
            out.append(l)
        else:
            raise ValueError(f'cannot broadcast {lhs} with {rhs}')
    out.reverse()
    return out

def py_strides(shape):
    strides = [0] * len(shape)
    stride = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = stride
        stride *= shape[i]
    return strides

def py_unravel(flat, shape, strides):
    if not shape:
        return []
    rem = flat
    out = [0] * len(shape)
    for i, (dim, stride) in enumerate(zip(shape, strides)):
        if dim == 0 or stride == 0:
            out[i] = 0
        else:
            out[i] = (rem // stride) % dim
            rem = rem % stride
    return out

def py_ravel(multi, strides):
    return sum(idx * stride for idx, stride in zip(multi, strides))

def py_src_index(out_multi, src_shape, src_strides, out_nd):
    if not src_shape:
        return 0
    offset = out_nd - len(src_shape)
    flat = 0
    for axis, (dim, stride) in enumerate(zip(src_shape, src_strides)):
        out_axis = axis + offset
        src_i = 0 if dim == 1 else out_multi[out_axis]
        flat += src_i * stride
    return flat

def py_binary(lhs_vals, lhs_shape, rhs_vals, rhs_shape, op):
    out_shape = py_broadcast_shape(lhs_shape, rhs_shape)
    out_count = math.prod(out_shape) if out_shape else 1
    out_strides = py_strides(out_shape)
    lhs_strides = py_strides(lhs_shape)
    rhs_strides = py_strides(rhs_shape)
    out_vals = []
    for flat in range(out_count):
        multi = py_unravel(flat, out_shape, out_strides)
        li = py_src_index(multi, lhs_shape, lhs_strides, len(out_shape))
        ri = py_src_index(multi, rhs_shape, rhs_strides, len(out_shape))
        l = lhs_vals[li]
        r = rhs_vals[ri]
        if op == 'add':
            out_vals.append(l + r)
        elif op == 'sub':
            out_vals.append(l - r)
        elif op == 'mul':
            out_vals.append(l * r)
        elif op == 'div':
            if r == 0.0:
                if l == 0.0:
                    out_vals.append(float('nan'))
                elif l > 0:
                    out_vals.append(float('inf'))
                else:
                    out_vals.append(float('-inf'))
            else:
                out_vals.append(l / r)
        elif op == 'power':
            out_vals.append(l ** r)
        elif op == 'remainder':
            if r == 0.0:
                out_vals.append(float('nan'))
            else:
                out_vals.append(l - math.floor(l / r) * r)
        elif op == 'minimum':
            out_vals.append(min(l, r) if not (math.isnan(l) or math.isnan(r)) else float('nan'))
        elif op == 'maximum':
            out_vals.append(max(l, r) if not (math.isnan(l) or math.isnan(r)) else float('nan'))
        elif op == 'arctan2':
            out_vals.append(math.atan2(l, r))
        elif op == 'fmod':
            if r == 0.0:
                out_vals.append(float('nan'))
            else:
                out_vals.append(math.fmod(l, r))
        elif op == 'copysign':
            out_vals.append(math.copysign(l, r))
        elif op == 'fmax':
            if math.isnan(l):
                out_vals.append(r)
            elif math.isnan(r):
                out_vals.append(l)
            else:
                out_vals.append(max(l, r))
        elif op == 'fmin':
            if math.isnan(l):
                out_vals.append(r)
            elif math.isnan(r):
                out_vals.append(l)
            else:
                out_vals.append(min(l, r))
        elif op == 'heaviside':
            if math.isnan(l):
                out_vals.append(float('nan'))
            elif l < 0:
                out_vals.append(0.0)
            elif l == 0:
                out_vals.append(r)
            else:
                out_vals.append(1.0)
        elif op == 'nextafter':
            out_vals.append(math.nextafter(l, r))
        elif op == 'logical_and':
            out_vals.append(1.0 if (l != 0.0 and r != 0.0) else 0.0)
        elif op == 'logical_or':
            out_vals.append(1.0 if (l != 0.0 or r != 0.0) else 0.0)
        elif op == 'logical_xor':
            out_vals.append(1.0 if ((l != 0.0) != (r != 0.0)) else 0.0)
        elif op == 'bitwise_and':
            out_vals.append(float(int(l) & int(r)))
        elif op == 'bitwise_or':
            out_vals.append(float(int(l) | int(r)))
        elif op == 'bitwise_xor':
            out_vals.append(float(int(l) ^ int(r)))
        elif op == 'equal':
            out_vals.append(1.0 if l == r else 0.0)
        elif op == 'not_equal':
            out_vals.append(1.0 if l != r else 0.0)
        elif op == 'less':
            out_vals.append(1.0 if l < r else 0.0)
        elif op == 'less_equal':
            out_vals.append(1.0 if l <= r else 0.0)
        elif op == 'greater':
            out_vals.append(1.0 if l > r else 0.0)
        elif op == 'greater_equal':
            out_vals.append(1.0 if l >= r else 0.0)
        elif op == 'hypot':
            out_vals.append(math.hypot(l, r))
        elif op == 'logaddexp':
            mx = max(l, r)
            mn = min(l, r)
            if mx == float('inf'):
                out_vals.append(float('inf'))
            elif math.isnan(l) or math.isnan(r):
                out_vals.append(float('nan'))
            else:
                out_vals.append(mx + math.log1p(math.exp(mn - mx)))
        elif op == 'logaddexp2':
            mx = max(l, r)
            mn = min(l, r)
            if mx == float('inf'):
                out_vals.append(float('inf'))
            elif math.isnan(l) or math.isnan(r):
                out_vals.append(float('nan'))
            else:
                out_vals.append(mx + math.log1p(2.0 ** (mn - mx)) / math.log(2.0))
        elif op == 'ldexp':
            out_vals.append(math.ldexp(l, int(r)))
        elif op == 'floor_divide':
            if r == 0.0:
                if l == 0.0:
                    out_vals.append(float('nan'))
                else:
                    out_vals.append(math.copysign(float('inf'), l))
            else:
                out_vals.append(math.floor(l / r))
        elif op == 'float_power':
            out_vals.append(l ** r)
        else:
            raise ValueError(f'unsupported op {op}')
    return out_shape, out_vals

def py_reduced_shape(shape, axis, keepdims):
    if keepdims:
        return [1 if i == axis else dim for i, dim in enumerate(shape)]
    return [dim for i, dim in enumerate(shape) if i != axis]

def py_sum(vals, shape, axis, keepdims):
    if axis is None:
        out_shape = [1] * len(shape) if keepdims else []
        return out_shape, [float(sum(vals))]

    raw_axis = axis
    if axis < 0:
        axis += len(shape)
    if axis < 0 or axis >= len(shape):
        raise ValueError(f'axis {raw_axis} out of bounds for shape {shape}')

    in_strides = py_strides(shape)
    out_shape = py_reduced_shape(shape, axis, keepdims)
    out_strides = py_strides(out_shape)
    out_count = math.prod(out_shape) if out_shape else 1
    out = [0.0] * out_count

    for flat in range(len(vals)):
        multi = py_unravel(flat, shape, in_strides)
        out_multi = []
        for i, idx in enumerate(multi):
            if i == axis:
                if keepdims:
                    out_multi.append(0)
            else:
                out_multi.append(idx)
        out_flat = py_ravel(out_multi, out_strides) if out_multi else 0
        out[out_flat] += vals[flat]
    return out_shape, out

def py_reduce(vals, shape, axis, keepdims, op):
    if axis is None:
        if op == 'prod':
            result = 1.0
            for v in vals:
                result *= v
        elif op == 'min':
            result = float('inf')
            for v in vals:
                result = min(result, v)
        elif op == 'max':
            result = float('-inf')
            for v in vals:
                result = max(result, v)
        elif op == 'mean':
            result = float(sum(vals)) / len(vals)
        else:
            raise ValueError(f'unsupported reduce op: {op}')
        out_shape = [1] * len(shape) if keepdims else []
        return out_shape, [result]

    raw_axis = axis
    if axis < 0:
        axis += len(shape)
    if axis < 0 or axis >= len(shape):
        raise ValueError(f'axis {raw_axis} out of bounds for shape {shape}')

    in_strides = py_strides(shape)
    out_shape = py_reduced_shape(shape, axis, keepdims)
    out_strides = py_strides(out_shape)
    out_count = math.prod(out_shape) if out_shape else 1

    if op == 'prod':
        out = [1.0] * out_count
    elif op == 'min':
        out = [float('inf')] * out_count
    elif op == 'max':
        out = [float('-inf')] * out_count
    elif op == 'mean':
        out = [0.0] * out_count
    else:
        raise ValueError(f'unsupported reduce op: {op}')

    for flat in range(len(vals)):
        multi = py_unravel(flat, shape, in_strides)
        out_multi = []
        for i, idx in enumerate(multi):
            if i == axis:
                if keepdims:
                    out_multi.append(0)
            else:
                out_multi.append(idx)
        out_flat = py_ravel(out_multi, out_strides) if out_multi else 0
        if op == 'prod':
            out[out_flat] *= vals[flat]
        elif op == 'min':
            out[out_flat] = min(out[out_flat], vals[flat])
        elif op == 'max':
            out[out_flat] = max(out[out_flat], vals[flat])
        elif op == 'mean':
            out[out_flat] += vals[flat]

    if op == 'mean':
        axis_len = shape[axis]
        out = [v / axis_len for v in out]

    return out_shape, out

def py_cumsum(vals, shape, axis):
    if axis is None:
        out = []
        acc = 0.0
        for v in vals:
            acc += v
            out.append(acc)
        return [len(out)], out
    raw_axis = axis
    if axis < 0:
        axis += len(shape)
    if axis < 0 or axis >= len(shape):
        raise ValueError(f'axis {raw_axis} out of bounds for shape {shape}')
    axis_len = shape[axis]
    inner = math.prod(shape[axis+1:]) if axis+1 < len(shape) else 1
    outer = math.prod(shape[:axis]) if axis > 0 else 1
    out = list(vals)
    for oi in range(outer):
        base = oi * axis_len * inner
        for ii in range(inner):
            acc = 0.0
            offset = base + ii
            for _ in range(axis_len):
                acc += out[offset]
                out[offset] = acc
                offset += inner
    return list(shape), out

def py_cumprod(vals, shape, axis):
    if axis is None:
        out = []
        acc = 1.0
        for v in vals:
            acc *= v
            out.append(acc)
        return [len(out)], out
    raw_axis = axis
    if axis < 0:
        axis += len(shape)
    if axis < 0 or axis >= len(shape):
        raise ValueError(f'axis {raw_axis} out of bounds for shape {shape}')
    axis_len = shape[axis]
    inner = math.prod(shape[axis+1:]) if axis+1 < len(shape) else 1
    outer = math.prod(shape[:axis]) if axis > 0 else 1
    out = list(vals)
    for oi in range(outer):
        base = oi * axis_len * inner
        for ii in range(inner):
            acc = 1.0
            offset = base + ii
            for _ in range(axis_len):
                acc *= out[offset]
                out[offset] = acc
                offset += inner
    return list(shape), out

def py_where(cond_vals, cond_shape, x_vals, x_shape, y_vals, y_shape):
    out_shape = py_broadcast_shape(cond_shape, x_shape)
    out_shape = py_broadcast_shape(out_shape, y_shape)
    out_count = math.prod(out_shape) if out_shape else 1
    out_strides = py_strides(out_shape)
    cond_strides = py_strides(cond_shape)
    x_strides = py_strides(x_shape)
    y_strides = py_strides(y_shape)
    out_vals = []
    for flat in range(out_count):
        multi = py_unravel(flat, out_shape, out_strides)
        ci = py_src_index(multi, cond_shape, cond_strides, len(out_shape))
        xi = py_src_index(multi, x_shape, x_strides, len(out_shape))
        yi = py_src_index(multi, y_shape, y_strides, len(out_shape))
        out_vals.append(x_vals[xi] if cond_vals[ci] != 0.0 else y_vals[yi])
    return out_shape, out_vals

def py_sort(vals, shape, axis):
    if axis is None:
        out = sorted(vals)
        return [len(out)], out

    raw_axis = axis
    if axis < 0:
        axis += len(shape)
    if axis < 0 or axis >= len(shape):
        raise ValueError(f'axis {raw_axis} out of bounds for shape {shape}')

    axis_len = shape[axis]
    if axis_len <= 1:
        return list(shape), list(vals)

    inner = math.prod(shape[axis+1:]) if axis + 1 < len(shape) else 1
    outer = math.prod(shape[:axis]) if axis > 0 else 1
    out = list(vals)
    for oi in range(outer):
        base = oi * axis_len * inner
        for ii in range(inner):
            lane = []
            for k in range(axis_len):
                lane.append(out[base + k * inner + ii])
            lane.sort()
            for k in range(axis_len):
                out[base + k * inner + ii] = lane[k]
    return list(shape), out

def py_argsort(vals, shape, axis):
    if axis is None:
        indices = list(range(len(vals)))
        indices.sort(key=lambda idx: vals[idx])
        return [len(indices)], [float(idx) for idx in indices]

    raw_axis = axis
    if axis < 0:
        axis += len(shape)
    if axis < 0 or axis >= len(shape):
        raise ValueError(f'axis {raw_axis} out of bounds for shape {shape}')

    axis_len = shape[axis]
    inner = math.prod(shape[axis+1:]) if axis + 1 < len(shape) else 1
    outer = math.prod(shape[:axis]) if axis > 0 else 1
    out = [0.0] * len(vals)
    for oi in range(outer):
        base = oi * axis_len * inner
        for ii in range(inner):
            idx_lane = list(range(axis_len))
            idx_lane.sort(key=lambda idx: vals[base + idx * inner + ii])
            for k in range(axis_len):
                out[base + k * inner + ii] = float(idx_lane[k])
    return list(shape), out

def py_concat2(lhs_vals, lhs_shape, rhs_vals, rhs_shape, axis):
    if len(lhs_shape) != len(rhs_shape):
        raise ValueError(f'rank mismatch lhs={lhs_shape} rhs={rhs_shape}')
    raw_axis = axis
    if axis < 0:
        axis += len(lhs_shape)
    if axis < 0 or axis >= len(lhs_shape):
        raise ValueError(f'axis {raw_axis} out of bounds for shape {lhs_shape}')

    for dim, (l, r) in enumerate(zip(lhs_shape, rhs_shape)):
        if dim != axis and l != r:
            raise ValueError(f'cannot concatenate {lhs_shape} with {rhs_shape}')

    out_shape = list(lhs_shape)
    out_shape[axis] = lhs_shape[axis] + rhs_shape[axis]
    out_count = math.prod(out_shape) if out_shape else 1
    inner = math.prod(lhs_shape[axis+1:]) if axis + 1 < len(lhs_shape) else 1
    outer = math.prod(lhs_shape[:axis]) if axis > 0 else 1
    out = [0.0] * out_count

    for oi in range(outer):
        write_offset = oi * out_shape[axis] * inner
        lhs_base = oi * lhs_shape[axis] * inner
        rhs_base = oi * rhs_shape[axis] * inner
        for k in range(lhs_shape[axis]):
            for ii in range(inner):
                out[write_offset + k * inner + ii] = lhs_vals[lhs_base + k * inner + ii]
        write_offset += lhs_shape[axis] * inner
        for k in range(rhs_shape[axis]):
            for ii in range(inner):
                out[write_offset + k * inner + ii] = rhs_vals[rhs_base + k * inner + ii]

    return out_shape, out

def py_stack2(lhs_vals, lhs_shape, rhs_vals, rhs_shape, axis):
    if lhs_shape != rhs_shape:
        raise ValueError(f'stack requires equal shapes lhs={lhs_shape} rhs={rhs_shape}')

    result_ndim = len(lhs_shape) + 1
    raw_axis = axis
    if axis < 0:
        axis += result_ndim
    if axis < 0 or axis >= result_ndim:
        raise ValueError(f'axis {raw_axis} out of bounds for stacked ndim={result_ndim}')

    expanded_shape = list(lhs_shape)
    expanded_shape.insert(axis, 1)
    return py_concat2(lhs_vals, expanded_shape, rhs_vals, expanded_shape, axis)

def py_searchsorted(sorted_vals, sorted_shape, probe_vals, probe_shape):
    if len(sorted_shape) != 1:
        raise ValueError(f'searchsorted expects 1-D sorted input, got shape {sorted_shape}')
    out = [float(bisect.bisect_left(sorted_vals, needle)) for needle in probe_vals]
    return list(probe_shape), out

def py_var_std(vals, shape, axis, keepdims, ddof, emit_std):
    ddof = int(ddof)
    if axis is None:
        count = len(vals)
        if count - ddof <= 0:
            raise ValueError(f'ddof {ddof} >= sample size {count}')
        mean = sum(vals) / count
        var = sum((v - mean) ** 2 for v in vals) / (count - ddof)
        out_shape = [1] * len(shape) if keepdims else []
        out_value = math.sqrt(var) if emit_std else var
        return out_shape, [out_value]

    raw_axis = axis
    if axis < 0:
        axis += len(shape)
    if axis < 0 or axis >= len(shape):
        raise ValueError(f'axis {raw_axis} out of bounds for shape {shape}')

    axis_len = shape[axis]
    if axis_len - ddof <= 0:
        raise ValueError(f'ddof {ddof} >= axis length {axis_len}')

    in_strides = py_strides(shape)
    out_shape = py_reduced_shape(shape, axis, keepdims)
    out_strides = py_strides(out_shape)
    out_count = math.prod(out_shape) if out_shape else 1

    sums = [0.0] * out_count
    sums_sq = [0.0] * out_count
    counts = [0] * out_count

    for flat in range(len(vals)):
        multi = py_unravel(flat, shape, in_strides)
        out_multi = []
        for i, idx in enumerate(multi):
            if i == axis:
                if keepdims:
                    out_multi.append(0)
            else:
                out_multi.append(idx)
        out_flat = py_ravel(out_multi, out_strides) if out_multi else 0
        value = vals[flat]
        sums[out_flat] += value
        sums_sq[out_flat] += value * value
        counts[out_flat] += 1

    out = []
    for i in range(out_count):
        count = counts[i]
        if count - ddof <= 0:
            raise ValueError(f'ddof {ddof} >= reduction size {count}')
        mean = sums[i] / count
        var = (sums_sq[i] - count * mean * mean) / (count - ddof)
        if var < 0.0 and abs(var) < 1e-15:
            var = 0.0
        out.append(math.sqrt(var) if emit_std else var)

    return out_shape, out

def py_arg_reduce(vals, shape, axis, find_min):
    if axis is None:
        if not vals:
            raise ValueError('arg-reduction of empty sequence')
        best_idx = 0
        best_val = vals[0]
        for idx in range(1, len(vals)):
            candidate = vals[idx]
            if (candidate < best_val) if find_min else (candidate > best_val):
                best_val = candidate
                best_idx = idx
        return [], [float(best_idx)]

    raw_axis = axis
    if axis < 0:
        axis += len(shape)
    if axis < 0 or axis >= len(shape):
        raise ValueError(f'axis {raw_axis} out of bounds for shape {shape}')

    axis_len = shape[axis]
    if axis_len == 0:
        raise ValueError('arg-reduction axis has length 0')

    inner = math.prod(shape[axis+1:]) if axis + 1 < len(shape) else 1
    outer = math.prod(shape[:axis]) if axis > 0 else 1
    out_shape = py_reduced_shape(shape, axis, False)
    out_vals = []

    for oi in range(outer):
        base = oi * axis_len * inner
        for ii in range(inner):
            best_idx = 0
            best_val = vals[base + ii]
            offset = base + ii + inner
            for k in range(1, axis_len):
                candidate = vals[offset]
                if (candidate < best_val) if find_min else (candidate > best_val):
                    best_val = candidate
                    best_idx = k
                offset += inner
            out_vals.append(float(best_idx))

    return out_shape, out_vals

def normalize_dtype_name(name):
    aliases = {
        'f64': 'float64',
        'f32': 'float32',
        'i64': 'int64',
        'i32': 'int32',
        'u64': 'uint64',
        'u32': 'uint32',
        'bool': 'bool_',
    }
    if name is None:
        return 'float64'
    key = str(name).strip().lower()
    return aliases.get(key, name)

def normalize_fallback_dtype(name):
    aliases = {
        'bool': 'bool',
        'bool_': 'bool',
        'i8': 'i8',
        'int8': 'i8',
        'i16': 'i16',
        'int16': 'i16',
        'i32': 'i32',
        'int32': 'i32',
        'i64': 'i64',
        'int64': 'i64',
        'u8': 'u8',
        'uint8': 'u8',
        'u16': 'u16',
        'uint16': 'u16',
        'u32': 'u32',
        'uint32': 'u32',
        'u64': 'u64',
        'uint64': 'u64',
        'f32': 'f32',
        'float32': 'f32',
        'f64': 'f64',
        'float64': 'f64',
    }
    if name is None:
        return 'f64'
    key = str(name).strip().lower()
    return aliases.get(key, 'f64')

def fallback_promote(lhs_dtype, rhs_dtype):
    lhs = normalize_fallback_dtype(lhs_dtype)
    rhs = normalize_fallback_dtype(rhs_dtype)

    if lhs == 'bool':
        return rhs
    if rhs == 'bool':
        return lhs
    if lhs == rhs:
        return lhs
    if (lhs == 'i32' and rhs == 'i64') or (lhs == 'i64' and rhs == 'i32'):
        return 'i64'
    if lhs == 'f64' or rhs == 'f64':
        return 'f64'
    if (lhs == 'f32' and rhs == 'f32'):
        return 'f32'
    if (lhs == 'f32' and rhs == 'i32') or (lhs == 'i32' and rhs == 'f32'):
        return 'f64'
    if (lhs == 'f32' and rhs == 'i64') or (lhs == 'i64' and rhs == 'f32'):
        return 'f64'
    return 'f64'

def fallback_sum_prod_dtype(dtype):
    dt = normalize_fallback_dtype(dtype)
    if dt in ('bool', 'i8', 'i16', 'i32'):
        return 'i64'
    if dt in ('u8', 'u16', 'u32', 'u64'):
        return 'u64'
    return dt

def fallback_mean_dtype(dtype):
    dt = normalize_fallback_dtype(dtype)
    if dt in ('bool', 'i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64'):
        return 'f64'
    return dt

results = []

for case in cases:
    case_id = case['id']
    op = case['op']
    try:
        if np is not None:
            lhs_dtype = normalize_dtype_name(case.get('lhs_dtype', 'float64'))
            lhs = np.array(case['lhs_values'], dtype=lhs_dtype).reshape(tuple(case['lhs_shape']))

            if op in ('add', 'sub', 'mul', 'div', 'power', 'remainder', 'minimum', 'maximum', 'arctan2', 'fmod', 'copysign', 'fmax', 'fmin', 'heaviside', 'nextafter', 'logical_and', 'logical_or', 'logical_xor', 'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal', 'hypot', 'logaddexp', 'logaddexp2', 'ldexp', 'floor_divide', 'float_power', 'bitwise_and', 'bitwise_or', 'bitwise_xor'):
                rhs_dtype = normalize_dtype_name(case.get('rhs_dtype', 'float64'))
                rhs = np.array(case['rhs_values'], dtype=rhs_dtype).reshape(tuple(case['rhs_shape']))
                if op == 'add':
                    out = lhs + rhs
                elif op == 'sub':
                    out = lhs - rhs
                elif op == 'mul':
                    out = lhs * rhs
                elif op == 'div':
                    out = lhs / rhs
                elif op == 'power':
                    out = np.power(lhs, rhs)
                elif op == 'remainder':
                    out = np.remainder(lhs, rhs)
                elif op == 'minimum':
                    out = np.minimum(lhs, rhs)
                elif op == 'maximum':
                    out = np.maximum(lhs, rhs)
                elif op == 'arctan2':
                    out = np.arctan2(lhs, rhs)
                elif op == 'fmod':
                    out = np.fmod(lhs, rhs)
                elif op == 'copysign':
                    out = np.copysign(lhs, rhs)
                elif op == 'fmax':
                    out = np.fmax(lhs, rhs)
                elif op == 'fmin':
                    out = np.fmin(lhs, rhs)
                elif op == 'heaviside':
                    out = np.heaviside(lhs, rhs)
                elif op == 'nextafter':
                    out = np.nextafter(lhs, rhs)
                elif op == 'logical_and':
                    out = np.logical_and(lhs, rhs)
                elif op == 'logical_or':
                    out = np.logical_or(lhs, rhs)
                elif op == 'logical_xor':
                    out = np.logical_xor(lhs, rhs)
                elif op == 'equal':
                    out = np.equal(lhs, rhs)
                elif op == 'not_equal':
                    out = np.not_equal(lhs, rhs)
                elif op == 'less':
                    out = np.less(lhs, rhs)
                elif op == 'less_equal':
                    out = np.less_equal(lhs, rhs)
                elif op == 'greater':
                    out = np.greater(lhs, rhs)
                elif op == 'greater_equal':
                    out = np.greater_equal(lhs, rhs)
                elif op == 'hypot':
                    out = np.hypot(lhs, rhs)
                elif op == 'logaddexp':
                    out = np.logaddexp(lhs, rhs)
                elif op == 'logaddexp2':
                    out = np.logaddexp2(lhs, rhs)
                elif op == 'ldexp':
                    out = np.ldexp(lhs, rhs.astype(np.int32))
                elif op == 'floor_divide':
                    out = np.floor_divide(lhs, rhs)
                elif op == 'float_power':
                    out = np.float_power(lhs, rhs)
                elif op == 'bitwise_and':
                    out = np.bitwise_and(lhs, rhs)
                elif op == 'bitwise_or':
                    out = np.bitwise_or(lhs, rhs)
                elif op == 'bitwise_xor':
                    out = np.bitwise_xor(lhs, rhs)
            elif op == 'sum':
                axis = case.get('axis')
                keepdims = bool(case.get('keepdims', False))
                out = lhs.sum(axis=axis, keepdims=keepdims)
            elif op == 'prod':
                axis = case.get('axis')
                keepdims = bool(case.get('keepdims', False))
                out = lhs.prod(axis=axis, keepdims=keepdims)
            elif op == 'min':
                axis = case.get('axis')
                keepdims = bool(case.get('keepdims', False))
                out = lhs.min(axis=axis, keepdims=keepdims)
            elif op == 'max':
                axis = case.get('axis')
                keepdims = bool(case.get('keepdims', False))
                out = lhs.max(axis=axis, keepdims=keepdims)
            elif op == 'mean':
                axis = case.get('axis')
                keepdims = bool(case.get('keepdims', False))
                out = lhs.mean(axis=axis, keepdims=keepdims)
            elif op == 'var':
                axis = case.get('axis')
                keepdims = bool(case.get('keepdims', False))
                ddof = int(case.get('ddof', 0) or 0)
                out = lhs.var(axis=axis, keepdims=keepdims, ddof=ddof)
            elif op == 'std':
                axis = case.get('axis')
                keepdims = bool(case.get('keepdims', False))
                ddof = int(case.get('ddof', 0) or 0)
                out = lhs.std(axis=axis, keepdims=keepdims, ddof=ddof)
            elif op == 'argmin':
                axis = case.get('axis')
                out = lhs.argmin(axis=axis)
            elif op == 'argmax':
                axis = case.get('axis')
                out = lhs.argmax(axis=axis)
            elif op == 'cumsum':
                axis = case.get('axis')
                out = np.cumsum(lhs, axis=axis)
            elif op == 'cumprod':
                axis = case.get('axis')
                out = np.cumprod(lhs, axis=axis)
            elif op == 'clip':
                clip_min = case.get('clip_min')
                clip_max = case.get('clip_max')
                out = np.clip(lhs, clip_min, clip_max)
            elif op == 'where':
                rhs_dtype = normalize_dtype_name(case.get('rhs_dtype', 'float64'))
                rhs = np.array(case['rhs_values'], dtype=rhs_dtype).reshape(tuple(case['rhs_shape']))
                third_dtype = normalize_dtype_name(case.get('third_dtype', 'float64'))
                third = np.array(case['third_values'], dtype=third_dtype).reshape(tuple(case['third_shape']))
                out = np.where(lhs, rhs, third)
            elif op == 'sort':
                axis = case.get('axis')
                out = np.sort(lhs, axis=axis)
            elif op == 'argsort':
                axis = case.get('axis')
                out = np.argsort(lhs, axis=axis)
            elif op == 'searchsorted':
                rhs_dtype = normalize_dtype_name(case.get('rhs_dtype', 'float64'))
                rhs = np.array(case['rhs_values'], dtype=rhs_dtype).reshape(tuple(case['rhs_shape']))
                out = np.searchsorted(lhs, rhs, side='left')
            elif op == 'concatenate':
                rhs_dtype = normalize_dtype_name(case.get('rhs_dtype', 'float64'))
                rhs = np.array(case['rhs_values'], dtype=rhs_dtype).reshape(tuple(case['rhs_shape']))
                axis = case.get('axis', 0)
                out = np.concatenate((lhs, rhs), axis=axis)
            elif op == 'stack':
                rhs_dtype = normalize_dtype_name(case.get('rhs_dtype', 'float64'))
                rhs = np.array(case['rhs_values'], dtype=rhs_dtype).reshape(tuple(case['rhs_shape']))
                axis = case.get('axis', 0)
                out = np.stack((lhs, rhs), axis=axis)
            elif op == 'abs':
                out = np.abs(lhs)
            elif op == 'negative':
                out = np.negative(lhs)
            elif op == 'sign':
                out = np.sign(lhs)
            elif op == 'sqrt':
                out = np.sqrt(lhs)
            elif op == 'square':
                out = np.square(lhs)
            elif op == 'exp':
                out = np.exp(lhs)
            elif op == 'log':
                out = np.log(lhs)
            elif op == 'log2':
                out = np.log2(lhs)
            elif op == 'log10':
                out = np.log10(lhs)
            elif op == 'sin':
                out = np.sin(lhs)
            elif op == 'cos':
                out = np.cos(lhs)
            elif op == 'tan':
                out = np.tan(lhs)
            elif op == 'floor':
                out = np.floor(lhs)
            elif op == 'ceil':
                out = np.ceil(lhs)
            elif op == 'round':
                out = np.round(lhs)
            elif op == 'reciprocal':
                out = np.reciprocal(lhs)
            elif op == 'sinh':
                out = np.sinh(lhs)
            elif op == 'cosh':
                out = np.cosh(lhs)
            elif op == 'tanh':
                out = np.tanh(lhs)
            elif op == 'arcsin':
                out = np.arcsin(lhs)
            elif op == 'arccos':
                out = np.arccos(lhs)
            elif op == 'arctan':
                out = np.arctan(lhs)
            elif op == 'cbrt':
                out = np.cbrt(lhs)
            elif op == 'expm1':
                out = np.expm1(lhs)
            elif op == 'log1p':
                out = np.log1p(lhs)
            elif op == 'degrees':
                out = np.degrees(lhs)
            elif op == 'radians':
                out = np.radians(lhs)
            elif op == 'rint':
                out = np.rint(lhs)
            elif op == 'trunc':
                out = np.trunc(lhs)
            elif op == 'positive':
                out = np.positive(lhs)
            elif op == 'spacing':
                out = np.spacing(lhs)
            elif op == 'logical_not':
                out = np.logical_not(lhs)
            elif op == 'isnan':
                out = np.isnan(lhs)
            elif op == 'isinf':
                out = np.isinf(lhs)
            elif op == 'isfinite':
                out = np.isfinite(lhs)
            elif op == 'signbit':
                out = np.signbit(lhs)
            elif op == 'exp2':
                out = np.exp2(lhs)
            elif op == 'fabs':
                out = np.fabs(lhs)
            elif op == 'arccosh':
                out = np.arccosh(lhs)
            elif op == 'arcsinh':
                out = np.arcsinh(lhs)
            elif op == 'arctanh':
                out = np.arctanh(lhs)
            else:
                raise ValueError(f'unsupported op: {op}')

            arr = np.asarray(out)
            values = arr.astype(np.float64, copy=False).reshape(-1).tolist()
            shape = list(arr.shape)
            dtype = str(arr.dtype)
        else:
            lhs_shape = case['lhs_shape']
            lhs_vals = [float(v) for v in case['lhs_values']]
            lhs_dtype = normalize_fallback_dtype(case.get('lhs_dtype', 'f64'))
            if op in ('add', 'sub', 'mul', 'div', 'power', 'remainder', 'minimum', 'maximum', 'arctan2', 'fmod', 'copysign', 'fmax', 'fmin', 'heaviside', 'nextafter', 'logical_and', 'logical_or', 'logical_xor', 'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal', 'hypot', 'logaddexp', 'logaddexp2', 'ldexp', 'floor_divide', 'float_power', 'bitwise_and', 'bitwise_or', 'bitwise_xor'):
                rhs_shape = case['rhs_shape']
                rhs_vals = [float(v) for v in case['rhs_values']]
                rhs_dtype = normalize_fallback_dtype(case.get('rhs_dtype', 'f64'))
                shape, values = py_binary(lhs_vals, lhs_shape, rhs_vals, rhs_shape, op)
                dtype = 'bool' if op in ('logical_and', 'logical_or', 'logical_xor', 'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal') else fallback_promote(lhs_dtype, rhs_dtype)
            elif op == 'sum':
                axis = case.get('axis')
                keepdims = bool(case.get('keepdims', False))
                shape, values = py_sum(lhs_vals, lhs_shape, axis, keepdims)
                dtype = fallback_sum_prod_dtype(lhs_dtype)
            elif op == 'prod':
                axis = case.get('axis')
                keepdims = bool(case.get('keepdims', False))
                shape, values = py_reduce(lhs_vals, lhs_shape, axis, keepdims, op)
                dtype = fallback_sum_prod_dtype(lhs_dtype)
            elif op in ('min', 'max'):
                axis = case.get('axis')
                keepdims = bool(case.get('keepdims', False))
                shape, values = py_reduce(lhs_vals, lhs_shape, axis, keepdims, op)
                dtype = lhs_dtype
            elif op == 'mean':
                axis = case.get('axis')
                keepdims = bool(case.get('keepdims', False))
                shape, values = py_reduce(lhs_vals, lhs_shape, axis, keepdims, op)
                dtype = fallback_mean_dtype(lhs_dtype)
            elif op == 'var':
                axis = case.get('axis')
                keepdims = bool(case.get('keepdims', False))
                ddof = int(case.get('ddof', 0) or 0)
                shape, values = py_var_std(lhs_vals, lhs_shape, axis, keepdims, ddof, False)
                dtype = 'f64'
            elif op == 'std':
                axis = case.get('axis')
                keepdims = bool(case.get('keepdims', False))
                ddof = int(case.get('ddof', 0) or 0)
                shape, values = py_var_std(lhs_vals, lhs_shape, axis, keepdims, ddof, True)
                dtype = 'f64'
            elif op == 'argmin':
                axis = case.get('axis')
                shape, values = py_arg_reduce(lhs_vals, lhs_shape, axis, True)
                dtype = 'i64'
            elif op == 'argmax':
                axis = case.get('axis')
                shape, values = py_arg_reduce(lhs_vals, lhs_shape, axis, False)
                dtype = 'i64'
            elif op == 'abs':
                shape = lhs_shape
                values = [abs(v) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'negative':
                shape = lhs_shape
                values = [-v for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'sign':
                shape = lhs_shape
                def _sign(v):
                    if v != v:
                        return float('nan')
                    return 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)
                values = [_sign(v) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'sqrt':
                import cmath
                shape = lhs_shape
                values = [cmath.sqrt(v).real if v >= 0 else float('nan') for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'square':
                shape = lhs_shape
                values = [v * v for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'exp':
                shape = lhs_shape
                values = [math.exp(v) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'log':
                shape = lhs_shape
                values = [math.log(v) if v > 0 else (float('-inf') if v == 0 else float('nan')) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'log2':
                shape = lhs_shape
                values = [math.log2(v) if v > 0 else (float('-inf') if v == 0 else float('nan')) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'log10':
                shape = lhs_shape
                values = [math.log10(v) if v > 0 else (float('-inf') if v == 0 else float('nan')) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'sin':
                shape = lhs_shape
                values = [math.sin(v) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'cos':
                shape = lhs_shape
                values = [math.cos(v) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'tan':
                shape = lhs_shape
                values = [math.tan(v) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'floor':
                shape = lhs_shape
                values = [math.floor(v) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'ceil':
                shape = lhs_shape
                values = [math.ceil(v) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'round':
                shape = lhs_shape
                values = [round(v) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'reciprocal':
                shape = lhs_shape
                values = [1.0 / v if v != 0 else float('inf') for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'sinh':
                shape = lhs_shape
                values = [math.sinh(v) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'cosh':
                shape = lhs_shape
                values = [math.cosh(v) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'tanh':
                shape = lhs_shape
                values = [math.tanh(v) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'arcsin':
                shape = lhs_shape
                values = [math.asin(v) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'arccos':
                shape = lhs_shape
                values = [math.acos(v) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'arctan':
                shape = lhs_shape
                values = [math.atan(v) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'cbrt':
                shape = lhs_shape
                values = [math.copysign(abs(v) ** (1.0/3.0), v) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'expm1':
                shape = lhs_shape
                values = [math.expm1(v) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'log1p':
                shape = lhs_shape
                values = [math.log1p(v) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'degrees':
                shape = lhs_shape
                values = [math.degrees(v) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'radians':
                shape = lhs_shape
                values = [math.radians(v) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'rint':
                shape = lhs_shape
                values = [round(v) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'trunc':
                shape = lhs_shape
                values = [math.trunc(v) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'positive':
                shape = lhs_shape
                values = [+v for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'spacing':
                shape = lhs_shape
                values = [math.nextafter(abs(v), float('inf')) - abs(v) if not (math.isnan(v) or math.isinf(v)) else float('nan') for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'logical_not':
                shape = lhs_shape
                values = [1.0 if v == 0.0 else 0.0 for v in lhs_vals]
                dtype = 'bool'
            elif op == 'isnan':
                shape = lhs_shape
                values = [1.0 if math.isnan(v) else 0.0 for v in lhs_vals]
                dtype = 'bool'
            elif op == 'isinf':
                shape = lhs_shape
                values = [1.0 if math.isinf(v) else 0.0 for v in lhs_vals]
                dtype = 'bool'
            elif op == 'isfinite':
                shape = lhs_shape
                values = [1.0 if math.isfinite(v) else 0.0 for v in lhs_vals]
                dtype = 'bool'
            elif op == 'signbit':
                shape = lhs_shape
                values = [1.0 if math.copysign(1.0, v) < 0 else 0.0 for v in lhs_vals]
                dtype = 'bool'
            elif op == 'exp2':
                shape = lhs_shape
                values = [2.0 ** v for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'fabs':
                shape = lhs_shape
                values = [abs(v) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'arccosh':
                shape = lhs_shape
                values = [math.acosh(v) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'arcsinh':
                shape = lhs_shape
                values = [math.asinh(v) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'arctanh':
                shape = lhs_shape
                values = [math.atanh(v) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'cumsum':
                axis = case.get('axis')
                shape, values = py_cumsum(lhs_vals, lhs_shape, axis)
                dtype = lhs_dtype
            elif op == 'cumprod':
                axis = case.get('axis')
                shape, values = py_cumprod(lhs_vals, lhs_shape, axis)
                dtype = lhs_dtype
            elif op == 'clip':
                clip_min = case.get('clip_min')
                clip_max = case.get('clip_max')
                shape = lhs_shape
                lo = clip_min if clip_min is not None else float('-inf')
                hi = clip_max if clip_max is not None else float('inf')
                values = [max(lo, min(hi, v)) for v in lhs_vals]
                dtype = lhs_dtype
            elif op == 'where':
                x_shape = case['rhs_shape']
                x_vals = [float(v) for v in case['rhs_values']]
                x_dtype = normalize_fallback_dtype(case.get('rhs_dtype', 'f64'))
                y_shape = case['third_shape']
                y_vals = [float(v) for v in case['third_values']]
                y_dtype = normalize_fallback_dtype(case.get('third_dtype', 'f64'))
                shape, values = py_where(lhs_vals, lhs_shape, x_vals, x_shape, y_vals, y_shape)
                dtype = fallback_promote(x_dtype, y_dtype)
            elif op == 'sort':
                axis = case.get('axis')
                shape, values = py_sort(lhs_vals, lhs_shape, axis)
                dtype = lhs_dtype
            elif op == 'argsort':
                axis = case.get('axis')
                shape, values = py_argsort(lhs_vals, lhs_shape, axis)
                dtype = 'i64'
            elif op == 'searchsorted':
                rhs_shape = case['rhs_shape']
                rhs_vals = [float(v) for v in case['rhs_values']]
                shape, values = py_searchsorted(lhs_vals, lhs_shape, rhs_vals, rhs_shape)
                dtype = 'i64'
            elif op == 'concatenate':
                rhs_shape = case['rhs_shape']
                rhs_vals = [float(v) for v in case['rhs_values']]
                rhs_dtype = normalize_fallback_dtype(case.get('rhs_dtype', 'f64'))
                axis = case.get('axis', 0)
                shape, values = py_concat2(lhs_vals, lhs_shape, rhs_vals, rhs_shape, axis)
                dtype = fallback_promote(lhs_dtype, rhs_dtype)
            elif op == 'stack':
                rhs_shape = case['rhs_shape']
                rhs_vals = [float(v) for v in case['rhs_values']]
                rhs_dtype = normalize_fallback_dtype(case.get('rhs_dtype', 'f64'))
                axis = case.get('axis', 0)
                shape, values = py_stack2(lhs_vals, lhs_shape, rhs_vals, rhs_shape, axis)
                dtype = fallback_promote(lhs_dtype, rhs_dtype)
            else:
                raise ValueError(f'unsupported op: {op}')

        results.append({
            'id': case_id,
            'status': 'ok',
            'error': None,
            'shape': shape,
            'values': values,
            'dtype': dtype,
        })
    except Exception as exc:
        results.append({
            'id': case_id,
            'status': 'error',
            'error': str(exc),
            'shape': [],
            'values': [],
            'dtype': 'unknown',
        })

payload = {
    'schema_version': 1,
    'oracle_source': oracle_source,
    'generated_at_unix_ms': 0,
    'cases': results,
}

with open(output_path, 'w', encoding='utf-8') as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UFuncOperation {
    Add,
    Sub,
    Mul,
    Div,
    Power,
    Remainder,
    Minimum,
    Maximum,
    Sum,
    Prod,
    Min,
    Max,
    Mean,
    Var,
    Std,
    Argmin,
    Argmax,
    Abs,
    Negative,
    Sign,
    Sqrt,
    Square,
    Exp,
    Log,
    Log2,
    Log10,
    Sin,
    Cos,
    Tan,
    Floor,
    Ceil,
    Round,
    Reciprocal,
    Sinh,
    Cosh,
    Tanh,
    Arcsin,
    Arccos,
    Arctan,
    Arctan2,
    Cbrt,
    Expm1,
    Log1p,
    Degrees,
    Radians,
    Rint,
    Trunc,
    Fmod,
    Copysign,
    Fmax,
    Fmin,
    Positive,
    Spacing,
    Heaviside,
    Nextafter,
    LogicalNot,
    LogicalAnd,
    LogicalOr,
    LogicalXor,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Isnan,
    Isinf,
    Isfinite,
    Signbit,
    Hypot,
    Logaddexp,
    Logaddexp2,
    Ldexp,
    Exp2,
    Fabs,
    Arccosh,
    Arcsinh,
    Arctanh,
    FloorDivide,
    FloatPower,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    Where,
    Sort,
    Argsort,
    Searchsorted,
    Concatenate,
    Stack,
    Cumsum,
    Cumprod,
    Clip,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UFuncInputCase {
    pub id: String,
    pub op: UFuncOperation,
    pub lhs_shape: Vec<usize>,
    pub lhs_values: Vec<f64>,
    #[serde(default = "default_f64_dtype")]
    pub lhs_dtype: String,
    pub rhs_shape: Option<Vec<usize>>,
    pub rhs_values: Option<Vec<f64>>,
    pub rhs_dtype: Option<String>,
    pub axis: Option<isize>,
    pub keepdims: Option<bool>,
    pub ddof: Option<usize>,
    pub clip_min: Option<f64>,
    pub clip_max: Option<f64>,
    pub third_shape: Option<Vec<usize>>,
    pub third_values: Option<Vec<f64>>,
    pub third_dtype: Option<String>,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub mode: String,
    #[serde(default)]
    pub env_fingerprint: String,
    #[serde(default)]
    pub artifact_refs: Vec<String>,
    #[serde(default)]
    pub reason_code: String,
    #[serde(default)]
    pub expected_reason_code: String,
    #[serde(default)]
    pub expected_error_contains: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UFuncOracleCase {
    pub id: String,
    pub status: String,
    pub error: Option<String>,
    pub shape: Vec<usize>,
    pub values: Vec<f64>,
    pub dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UFuncOracleCapture {
    pub schema_version: u8,
    pub oracle_source: String,
    pub generated_at_unix_ms: u128,
    pub cases: Vec<UFuncOracleCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UFuncDifferentialCaseResult {
    pub id: String,
    pub seed: u64,
    pub mode: String,
    pub env_fingerprint: String,
    pub artifact_refs: Vec<String>,
    pub pass: bool,
    pub max_abs_error: f64,
    pub expected_shape: Vec<usize>,
    pub actual_shape: Vec<usize>,
    pub expected_dtype: String,
    pub actual_dtype: String,
    pub expected_reason_code: String,
    pub actual_reason_code: String,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UFuncDifferentialReport {
    pub schema_version: u8,
    pub oracle_source: String,
    pub generated_at_unix_ms: u128,
    pub abs_tol: f64,
    pub rel_tol: f64,
    pub total_cases: usize,
    pub passed_cases: usize,
    pub failed_cases: usize,
    pub failures: Vec<UFuncDifferentialCaseResult>,
}

fn default_f64_dtype() -> String {
    "f64".to_string()
}

fn normalize_mode(raw: &str) -> String {
    if raw.trim().is_empty() {
        "strict".to_string()
    } else {
        raw.trim().to_string()
    }
}

fn normalize_env_fingerprint(raw: &str) -> String {
    if raw.trim().is_empty() {
        "unknown_env".to_string()
    } else {
        raw.trim().to_string()
    }
}

fn normalize_artifact_refs(mut refs: Vec<String>) -> Vec<String> {
    refs.retain(|entry| !entry.trim().is_empty());
    if refs.is_empty() {
        refs.push("crates/fnp-conformance/fixtures/ufunc_input_cases.json".to_string());
    }
    refs
}

fn classify_reason_code(op: UFuncOperation, detail: &str) -> String {
    let lowered = detail.to_lowercase();

    if lowered.contains("oracle case missing")
        || lowered.contains("unsupported oracle status")
        || lowered.contains("unknown metadata")
    {
        "ufunc_policy_unknown_metadata".to_string()
    } else if lowered.contains("signature")
        || lowered.contains("rhs_shape")
        || lowered.contains("rhs_values")
    {
        "ufunc_signature_parse_failed".to_string()
    } else if lowered.contains("dtype mismatch")
        || lowered.contains("unsupported dtype")
        || lowered.contains("type resolution")
    {
        "ufunc_type_resolution_invalid".to_string()
    } else if matches!(
        op,
        UFuncOperation::Sum
            | UFuncOperation::Prod
            | UFuncOperation::Min
            | UFuncOperation::Max
            | UFuncOperation::Mean
            | UFuncOperation::Var
            | UFuncOperation::Std
            | UFuncOperation::Argmin
            | UFuncOperation::Argmax
    ) && (lowered.contains("axis")
        || lowered.contains("keepdims")
        || lowered.contains("argmin")
        || lowered.contains("argmax")
        || lowered.contains("reduce var")
        || lowered.contains("reduce std")
        || lowered.contains("reduce sum"))
    {
        "ufunc_reduction_contract_violation".to_string()
    } else if lowered.contains("override") {
        "ufunc_override_precedence_violation".to_string()
    } else {
        "ufunc_dispatch_resolution_failed".to_string()
    }
}

fn resolve_expected_reason_code(input: &UFuncInputCase, fallback: &str) -> String {
    if !input.expected_reason_code.trim().is_empty() {
        input.expected_reason_code.trim().to_string()
    } else if !input.reason_code.trim().is_empty() {
        input.reason_code.trim().to_string()
    } else {
        fallback.to_string()
    }
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn resolve_oracle_python() -> String {
    std::env::var("FNP_ORACLE_PYTHON")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "python3".to_string())
}

pub fn load_input_cases(path: &Path) -> Result<Vec<UFuncInputCase>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    serde_json::from_str(&raw)
        .map_err(|err| format!("invalid input json {}: {err}", path.display()))
}

pub fn load_oracle_capture(path: &Path) -> Result<UFuncOracleCapture, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    serde_json::from_str(&raw)
        .map_err(|err| format!("invalid oracle json {}: {err}", path.display()))
}

pub fn write_oracle_capture(path: &Path, capture: &UFuncOracleCapture) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }

    let raw = serde_json::to_string_pretty(capture)
        .map_err(|err| format!("failed to serialize oracle capture: {err}"))?;
    fs::write(path, raw).map_err(|err| format!("failed writing {}: {err}", path.display()))
}

pub fn write_differential_report(
    path: &Path,
    report: &UFuncDifferentialReport,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }

    let raw = serde_json::to_string_pretty(report)
        .map_err(|err| format!("failed to serialize differential report: {err}"))?;
    fs::write(path, raw).map_err(|err| format!("failed writing {}: {err}", path.display()))
}

pub fn capture_numpy_oracle(
    input_path: &Path,
    output_path: &Path,
    legacy_oracle_root: &Path,
) -> Result<UFuncOracleCapture, String> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }

    let python = resolve_oracle_python();
    let output = Command::new(&python)
        .arg("-c")
        .arg(PY_CAPTURE_SCRIPT)
        .arg(input_path)
        .arg(output_path)
        .arg(legacy_oracle_root)
        .output()
        .map_err(|err| format!("failed to invoke oracle python '{}': {err}", python))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "python capture failed interpreter={} status={} stdout={} stderr={}",
            python,
            output.status,
            stdout.trim(),
            stderr.trim()
        ));
    }

    let mut capture = load_oracle_capture(output_path)?;
    capture.generated_at_unix_ms = now_unix_ms();
    write_oracle_capture(output_path, &capture)?;
    Ok(capture)
}

pub fn compare_against_oracle(
    input_path: &Path,
    oracle_path: &Path,
    abs_tol: f64,
    rel_tol: f64,
) -> Result<UFuncDifferentialReport, String> {
    let inputs = load_input_cases(input_path)?;
    let oracle = load_oracle_capture(oracle_path)?;

    let mut oracle_map: BTreeMap<&str, &UFuncOracleCase> = BTreeMap::new();
    for case in &oracle.cases {
        oracle_map.insert(case.id.as_str(), case);
    }

    let mut failures = Vec::new();
    let mut passed = 0usize;

    for input in &inputs {
        let mode = normalize_mode(&input.mode);
        let env_fingerprint = normalize_env_fingerprint(&input.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(input.artifact_refs.clone());

        let oracle_case = match oracle_map.get(input.id.as_str()) {
            Some(case) => *case,
            None => {
                let actual_reason_code = classify_reason_code(input.op, "oracle case missing");
                let expected_reason_code = resolve_expected_reason_code(input, &actual_reason_code);
                failures.push(UFuncDifferentialCaseResult {
                    id: input.id.clone(),
                    seed: input.seed,
                    mode: mode.clone(),
                    env_fingerprint: env_fingerprint.clone(),
                    artifact_refs: artifact_refs.clone(),
                    pass: false,
                    max_abs_error: f64::INFINITY,
                    expected_shape: Vec::new(),
                    actual_shape: Vec::new(),
                    expected_dtype: "missing".to_string(),
                    actual_dtype: "unknown".to_string(),
                    expected_reason_code,
                    actual_reason_code,
                    reason: Some("oracle case missing".to_string()),
                });
                continue;
            }
        };

        let outcome = execute_input_case(input);
        match (oracle_case.status.as_str(), outcome) {
            ("ok", Ok((actual_shape, actual_values, actual_dtype))) => {
                let expected_dtype_canonical = canonical_dtype_name(&oracle_case.dtype);
                let actual_dtype_canonical = canonical_dtype_name(&actual_dtype);
                if expected_dtype_canonical != actual_dtype_canonical {
                    let reason = format!(
                        "dtype mismatch expected={} actual={}",
                        oracle_case.dtype, actual_dtype
                    );
                    let actual_reason_code = classify_reason_code(input.op, &reason);
                    let expected_reason_code =
                        resolve_expected_reason_code(input, "ufunc_type_resolution_invalid");
                    failures.push(UFuncDifferentialCaseResult {
                        id: input.id.clone(),
                        seed: input.seed,
                        mode: mode.clone(),
                        env_fingerprint: env_fingerprint.clone(),
                        artifact_refs: artifact_refs.clone(),
                        pass: false,
                        max_abs_error: f64::INFINITY,
                        expected_shape: oracle_case.shape.clone(),
                        actual_shape,
                        expected_dtype: oracle_case.dtype.clone(),
                        actual_dtype,
                        expected_reason_code,
                        actual_reason_code,
                        reason: Some(reason),
                    });
                    continue;
                }

                let (pass, max_abs_error, reason) = compare_arrays(
                    &oracle_case.shape,
                    &oracle_case.values,
                    &actual_shape,
                    &actual_values,
                    abs_tol,
                    rel_tol,
                );

                if pass {
                    passed += 1;
                } else {
                    let reason_text = reason
                        .clone()
                        .unwrap_or_else(|| "unknown differential mismatch".to_string());
                    let actual_reason_code = classify_reason_code(input.op, &reason_text);
                    let expected_reason_code =
                        resolve_expected_reason_code(input, &actual_reason_code);
                    failures.push(UFuncDifferentialCaseResult {
                        id: input.id.clone(),
                        seed: input.seed,
                        mode: mode.clone(),
                        env_fingerprint: env_fingerprint.clone(),
                        artifact_refs: artifact_refs.clone(),
                        pass,
                        max_abs_error,
                        expected_shape: oracle_case.shape.clone(),
                        actual_shape,
                        expected_dtype: oracle_case.dtype.clone(),
                        actual_dtype,
                        expected_reason_code,
                        actual_reason_code,
                        reason,
                    });
                }
            }
            ("ok", Err(err)) => {
                let detail = format!("execution failed: {err}");
                let actual_reason_code = classify_reason_code(input.op, &detail);
                let expected_reason_code = resolve_expected_reason_code(input, &actual_reason_code);
                failures.push(UFuncDifferentialCaseResult {
                    id: input.id.clone(),
                    seed: input.seed,
                    mode: mode.clone(),
                    env_fingerprint: env_fingerprint.clone(),
                    artifact_refs: artifact_refs.clone(),
                    pass: false,
                    max_abs_error: f64::INFINITY,
                    expected_shape: oracle_case.shape.clone(),
                    actual_shape: Vec::new(),
                    expected_dtype: oracle_case.dtype.clone(),
                    actual_dtype: "error".to_string(),
                    expected_reason_code,
                    actual_reason_code,
                    reason: Some(detail),
                });
            }
            ("error", Ok((actual_shape, _, actual_dtype))) => {
                let oracle_error = oracle_case
                    .error
                    .clone()
                    .unwrap_or_else(|| "unknown error".to_string());
                let expected_reason_code = resolve_expected_reason_code(
                    input,
                    &classify_reason_code(input.op, &oracle_error),
                );
                failures.push(UFuncDifferentialCaseResult {
                    id: input.id.clone(),
                    seed: input.seed,
                    mode: mode.clone(),
                    env_fingerprint: env_fingerprint.clone(),
                    artifact_refs: artifact_refs.clone(),
                    pass: false,
                    max_abs_error: f64::INFINITY,
                    expected_shape: oracle_case.shape.clone(),
                    actual_shape,
                    expected_dtype: oracle_case.dtype.clone(),
                    actual_dtype,
                    expected_reason_code,
                    actual_reason_code: "ufunc_dispatch_resolution_failed".to_string(),
                    reason: Some(format!(
                        "oracle expected error '{}' but execution succeeded",
                        oracle_error
                    )),
                });
            }
            ("error", Err(err)) => {
                let oracle_error = oracle_case
                    .error
                    .clone()
                    .unwrap_or_else(|| "unknown error".to_string());
                let actual_reason_code = classify_reason_code(input.op, &err);
                let expected_reason_code = resolve_expected_reason_code(
                    input,
                    &classify_reason_code(input.op, &oracle_error),
                );

                let expected_error_contains = input.expected_error_contains.trim().to_lowercase();
                let actual_lower = err.to_lowercase();
                let error_match = if expected_error_contains.is_empty() {
                    true
                } else {
                    actual_lower.contains(&expected_error_contains)
                };
                let reason_match = expected_reason_code == actual_reason_code;

                if error_match && reason_match {
                    passed += 1;
                } else {
                    let detail = if expected_error_contains.is_empty() {
                        format!(
                            "oracle/local error disagreement: oracle='{}' actual='{}' expected_reason_code='{}' actual_reason_code='{}'",
                            oracle_error, err, expected_reason_code, actual_reason_code
                        )
                    } else {
                        format!(
                            "oracle/local error mismatch expected_error_contains='{}' oracle='{}' actual='{}' expected_reason_code='{}' actual_reason_code='{}'",
                            expected_error_contains,
                            oracle_error,
                            err,
                            expected_reason_code,
                            actual_reason_code
                        )
                    };
                    failures.push(UFuncDifferentialCaseResult {
                        id: input.id.clone(),
                        seed: input.seed,
                        mode: mode.clone(),
                        env_fingerprint: env_fingerprint.clone(),
                        artifact_refs: artifact_refs.clone(),
                        pass: false,
                        max_abs_error: f64::INFINITY,
                        expected_shape: oracle_case.shape.clone(),
                        actual_shape: Vec::new(),
                        expected_dtype: oracle_case.dtype.clone(),
                        actual_dtype: "error".to_string(),
                        expected_reason_code,
                        actual_reason_code,
                        reason: Some(detail),
                    });
                }
            }
            (status, _) => {
                let detail = format!("unsupported oracle status: {status}");
                let actual_reason_code = classify_reason_code(input.op, &detail);
                let expected_reason_code = resolve_expected_reason_code(input, &actual_reason_code);
                failures.push(UFuncDifferentialCaseResult {
                    id: input.id.clone(),
                    seed: input.seed,
                    mode,
                    env_fingerprint,
                    artifact_refs,
                    pass: false,
                    max_abs_error: f64::INFINITY,
                    expected_shape: oracle_case.shape.clone(),
                    actual_shape: Vec::new(),
                    expected_dtype: oracle_case.dtype.clone(),
                    actual_dtype: "unknown".to_string(),
                    expected_reason_code,
                    actual_reason_code,
                    reason: Some(detail),
                });
            }
        }
    }

    Ok(UFuncDifferentialReport {
        schema_version: 1,
        oracle_source: oracle.oracle_source,
        // Keep differential artifacts deterministic across repeated local/CI runs
        // when input/oracle fixtures are unchanged.
        generated_at_unix_ms: oracle.generated_at_unix_ms,
        abs_tol,
        rel_tol,
        total_cases: inputs.len(),
        passed_cases: passed,
        failed_cases: failures.len(),
        failures,
    })
}

pub fn execute_input_case(case: &UFuncInputCase) -> Result<(Vec<usize>, Vec<f64>, String), String> {
    let lhs_dtype = parse_dtype(&case.lhs_dtype)?;
    let lhs = UFuncArray::new(case.lhs_shape.clone(), case.lhs_values.clone(), lhs_dtype)
        .map_err(|err| format!("lhs array error: {err}"))?;

    let out = match case.op {
        UFuncOperation::Add
        | UFuncOperation::Sub
        | UFuncOperation::Mul
        | UFuncOperation::Div
        | UFuncOperation::Power
        | UFuncOperation::Remainder
        | UFuncOperation::Minimum
        | UFuncOperation::Maximum
        | UFuncOperation::Arctan2
        | UFuncOperation::Fmod
        | UFuncOperation::Copysign
        | UFuncOperation::Fmax
        | UFuncOperation::Fmin
        | UFuncOperation::Heaviside
        | UFuncOperation::Nextafter
        | UFuncOperation::LogicalAnd
        | UFuncOperation::LogicalOr
        | UFuncOperation::LogicalXor
        | UFuncOperation::Equal
        | UFuncOperation::NotEqual
        | UFuncOperation::Less
        | UFuncOperation::LessEqual
        | UFuncOperation::Greater
        | UFuncOperation::GreaterEqual
        | UFuncOperation::Hypot
        | UFuncOperation::Logaddexp
        | UFuncOperation::Logaddexp2
        | UFuncOperation::Ldexp
        | UFuncOperation::FloorDivide
        | UFuncOperation::FloatPower
        | UFuncOperation::BitwiseAnd
        | UFuncOperation::BitwiseOr
        | UFuncOperation::BitwiseXor => {
            let rhs_shape = case
                .rhs_shape
                .clone()
                .ok_or_else(|| "binary op requires rhs_shape".to_string())?;
            let rhs_values = case
                .rhs_values
                .clone()
                .ok_or_else(|| "binary op requires rhs_values".to_string())?;
            let rhs_dtype_name = case.rhs_dtype.as_deref().unwrap_or("f64");
            let rhs_dtype = parse_dtype(rhs_dtype_name)?;
            let rhs = UFuncArray::new(rhs_shape, rhs_values, rhs_dtype)
                .map_err(|err| format!("rhs array error: {err}"))?;

            let op = match case.op {
                UFuncOperation::Add => BinaryOp::Add,
                UFuncOperation::Sub => BinaryOp::Sub,
                UFuncOperation::Mul => BinaryOp::Mul,
                UFuncOperation::Div => BinaryOp::Div,
                UFuncOperation::Power => BinaryOp::Power,
                UFuncOperation::Remainder => BinaryOp::Remainder,
                UFuncOperation::Minimum => BinaryOp::Minimum,
                UFuncOperation::Maximum => BinaryOp::Maximum,
                UFuncOperation::Arctan2 => BinaryOp::Arctan2,
                UFuncOperation::Fmod => BinaryOp::Fmod,
                UFuncOperation::Copysign => BinaryOp::Copysign,
                UFuncOperation::Fmax => BinaryOp::Fmax,
                UFuncOperation::Fmin => BinaryOp::Fmin,
                UFuncOperation::Heaviside => BinaryOp::Heaviside,
                UFuncOperation::Nextafter => BinaryOp::Nextafter,
                UFuncOperation::LogicalAnd => BinaryOp::LogicalAnd,
                UFuncOperation::LogicalOr => BinaryOp::LogicalOr,
                UFuncOperation::LogicalXor => BinaryOp::LogicalXor,
                UFuncOperation::Equal => BinaryOp::Equal,
                UFuncOperation::NotEqual => BinaryOp::NotEqual,
                UFuncOperation::Less => BinaryOp::Less,
                UFuncOperation::LessEqual => BinaryOp::LessEqual,
                UFuncOperation::Greater => BinaryOp::Greater,
                UFuncOperation::GreaterEqual => BinaryOp::GreaterEqual,
                UFuncOperation::Hypot => BinaryOp::Hypot,
                UFuncOperation::Logaddexp => BinaryOp::Logaddexp,
                UFuncOperation::Logaddexp2 => BinaryOp::Logaddexp2,
                UFuncOperation::Ldexp => BinaryOp::Ldexp,
                UFuncOperation::FloorDivide => BinaryOp::FloorDivide,
                UFuncOperation::FloatPower => BinaryOp::FloatPower,
                UFuncOperation::BitwiseAnd => BinaryOp::BitwiseAnd,
                UFuncOperation::BitwiseOr => BinaryOp::BitwiseOr,
                UFuncOperation::BitwiseXor => BinaryOp::BitwiseXor,
                _ => unreachable!("handled above"),
            };

            lhs.elementwise_binary(&rhs, op)
                .map_err(|err| format!("binary op error: {err}"))?
        }
        UFuncOperation::Sum => {
            let keepdims = case.keepdims.unwrap_or(false);
            lhs.reduce_sum(case.axis, keepdims)
                .map_err(|err| format!("reduce sum error: {err}"))?
        }
        UFuncOperation::Prod => {
            let keepdims = case.keepdims.unwrap_or(false);
            lhs.reduce_prod(case.axis, keepdims)
                .map_err(|err| format!("reduce prod error: {err}"))?
        }
        UFuncOperation::Min => {
            let keepdims = case.keepdims.unwrap_or(false);
            lhs.reduce_min(case.axis, keepdims)
                .map_err(|err| format!("reduce min error: {err}"))?
        }
        UFuncOperation::Max => {
            let keepdims = case.keepdims.unwrap_or(false);
            lhs.reduce_max(case.axis, keepdims)
                .map_err(|err| format!("reduce max error: {err}"))?
        }
        UFuncOperation::Mean => {
            let keepdims = case.keepdims.unwrap_or(false);
            lhs.reduce_mean(case.axis, keepdims)
                .map_err(|err| format!("reduce mean error: {err}"))?
        }
        UFuncOperation::Var => {
            let keepdims = case.keepdims.unwrap_or(false);
            let ddof = case.ddof.unwrap_or(0);
            lhs.reduce_var(case.axis, keepdims, ddof)
                .map_err(|err| format!("reduce var error: {err}"))?
        }
        UFuncOperation::Std => {
            let keepdims = case.keepdims.unwrap_or(false);
            let ddof = case.ddof.unwrap_or(0);
            lhs.reduce_std(case.axis, keepdims, ddof)
                .map_err(|err| format!("reduce std error: {err}"))?
        }
        UFuncOperation::Argmin => lhs
            .reduce_argmin(case.axis)
            .map_err(|err| format!("reduce argmin error: {err}"))?,
        UFuncOperation::Argmax => lhs
            .reduce_argmax(case.axis)
            .map_err(|err| format!("reduce argmax error: {err}"))?,
        UFuncOperation::Cumsum => lhs
            .cumsum(case.axis)
            .map_err(|err| format!("cumsum error: {err}"))?,
        UFuncOperation::Cumprod => lhs
            .cumprod(case.axis)
            .map_err(|err| format!("cumprod error: {err}"))?,
        UFuncOperation::Clip => {
            let min_val = case.clip_min.unwrap_or(f64::NEG_INFINITY);
            let max_val = case.clip_max.unwrap_or(f64::INFINITY);
            lhs.clip(min_val, max_val)
        }
        UFuncOperation::Where => {
            let x_shape = case
                .rhs_shape
                .clone()
                .ok_or_else(|| "where requires rhs_shape for x".to_string())?;
            let x_values = case
                .rhs_values
                .clone()
                .ok_or_else(|| "where requires rhs_values for x".to_string())?;
            let x_dtype = parse_dtype(case.rhs_dtype.as_deref().unwrap_or("f64"))?;
            let x = UFuncArray::new(x_shape, x_values, x_dtype)
                .map_err(|err| format!("where x array error: {err}"))?;

            let y_shape = case
                .third_shape
                .clone()
                .ok_or_else(|| "where requires third_shape for y".to_string())?;
            let y_values = case
                .third_values
                .clone()
                .ok_or_else(|| "where requires third_values for y".to_string())?;
            let y_dtype = parse_dtype(case.third_dtype.as_deref().unwrap_or("f64"))?;
            let y = UFuncArray::new(y_shape, y_values, y_dtype)
                .map_err(|err| format!("where y array error: {err}"))?;

            UFuncArray::where_select(&lhs, &x, &y).map_err(|err| format!("where error: {err}"))?
        }
        UFuncOperation::Sort => lhs
            .sort(case.axis)
            .map_err(|err| format!("sort error: {err}"))?,
        UFuncOperation::Argsort => lhs
            .argsort(case.axis)
            .map_err(|err| format!("argsort error: {err}"))?,
        UFuncOperation::Searchsorted => {
            let probe_shape = case
                .rhs_shape
                .clone()
                .ok_or_else(|| "searchsorted requires rhs_shape".to_string())?;
            let probe_values = case
                .rhs_values
                .clone()
                .ok_or_else(|| "searchsorted requires rhs_values".to_string())?;
            let probe_dtype = parse_dtype(case.rhs_dtype.as_deref().unwrap_or("f64"))?;
            let probes = UFuncArray::new(probe_shape, probe_values, probe_dtype)
                .map_err(|err| format!("searchsorted probe error: {err}"))?;
            lhs.searchsorted(&probes)
                .map_err(|err| format!("searchsorted error: {err}"))?
        }
        UFuncOperation::Concatenate => {
            let rhs_shape = case
                .rhs_shape
                .clone()
                .ok_or_else(|| "concatenate requires rhs_shape".to_string())?;
            let rhs_values = case
                .rhs_values
                .clone()
                .ok_or_else(|| "concatenate requires rhs_values".to_string())?;
            let rhs_dtype = parse_dtype(case.rhs_dtype.as_deref().unwrap_or("f64"))?;
            let rhs = UFuncArray::new(rhs_shape, rhs_values, rhs_dtype)
                .map_err(|err| format!("concatenate rhs error: {err}"))?;
            UFuncArray::concatenate(&[&lhs, &rhs], case.axis.unwrap_or(0))
                .map_err(|err| format!("concatenate error: {err}"))?
        }
        UFuncOperation::Stack => {
            let rhs_shape = case
                .rhs_shape
                .clone()
                .ok_or_else(|| "stack requires rhs_shape".to_string())?;
            let rhs_values = case
                .rhs_values
                .clone()
                .ok_or_else(|| "stack requires rhs_values".to_string())?;
            let rhs_dtype = parse_dtype(case.rhs_dtype.as_deref().unwrap_or("f64"))?;
            let rhs = UFuncArray::new(rhs_shape, rhs_values, rhs_dtype)
                .map_err(|err| format!("stack rhs error: {err}"))?;
            UFuncArray::stack(&[&lhs, &rhs], case.axis.unwrap_or(0))
                .map_err(|err| format!("stack error: {err}"))?
        }
        UFuncOperation::Abs => lhs.elementwise_unary(UnaryOp::Abs),
        UFuncOperation::Negative => lhs.elementwise_unary(UnaryOp::Negative),
        UFuncOperation::Sign => lhs.elementwise_unary(UnaryOp::Sign),
        UFuncOperation::Sqrt => lhs.elementwise_unary(UnaryOp::Sqrt),
        UFuncOperation::Square => lhs.elementwise_unary(UnaryOp::Square),
        UFuncOperation::Exp => lhs.elementwise_unary(UnaryOp::Exp),
        UFuncOperation::Log => lhs.elementwise_unary(UnaryOp::Log),
        UFuncOperation::Log2 => lhs.elementwise_unary(UnaryOp::Log2),
        UFuncOperation::Log10 => lhs.elementwise_unary(UnaryOp::Log10),
        UFuncOperation::Sin => lhs.elementwise_unary(UnaryOp::Sin),
        UFuncOperation::Cos => lhs.elementwise_unary(UnaryOp::Cos),
        UFuncOperation::Tan => lhs.elementwise_unary(UnaryOp::Tan),
        UFuncOperation::Floor => lhs.elementwise_unary(UnaryOp::Floor),
        UFuncOperation::Ceil => lhs.elementwise_unary(UnaryOp::Ceil),
        UFuncOperation::Round => lhs.elementwise_unary(UnaryOp::Round),
        UFuncOperation::Reciprocal => lhs.elementwise_unary(UnaryOp::Reciprocal),
        UFuncOperation::Sinh => lhs.elementwise_unary(UnaryOp::Sinh),
        UFuncOperation::Cosh => lhs.elementwise_unary(UnaryOp::Cosh),
        UFuncOperation::Tanh => lhs.elementwise_unary(UnaryOp::Tanh),
        UFuncOperation::Arcsin => lhs.elementwise_unary(UnaryOp::Arcsin),
        UFuncOperation::Arccos => lhs.elementwise_unary(UnaryOp::Arccos),
        UFuncOperation::Arctan => lhs.elementwise_unary(UnaryOp::Arctan),
        UFuncOperation::Cbrt => lhs.elementwise_unary(UnaryOp::Cbrt),
        UFuncOperation::Expm1 => lhs.elementwise_unary(UnaryOp::Expm1),
        UFuncOperation::Log1p => lhs.elementwise_unary(UnaryOp::Log1p),
        UFuncOperation::Degrees => lhs.elementwise_unary(UnaryOp::Degrees),
        UFuncOperation::Radians => lhs.elementwise_unary(UnaryOp::Radians),
        UFuncOperation::Rint => lhs.elementwise_unary(UnaryOp::Rint),
        UFuncOperation::Trunc => lhs.elementwise_unary(UnaryOp::Trunc),
        UFuncOperation::Positive => lhs.elementwise_unary(UnaryOp::Positive),
        UFuncOperation::Spacing => lhs.elementwise_unary(UnaryOp::Spacing),
        UFuncOperation::LogicalNot => lhs.elementwise_unary(UnaryOp::LogicalNot),
        UFuncOperation::Isnan => lhs.elementwise_unary(UnaryOp::Isnan),
        UFuncOperation::Isinf => lhs.elementwise_unary(UnaryOp::Isinf),
        UFuncOperation::Isfinite => lhs.elementwise_unary(UnaryOp::Isfinite),
        UFuncOperation::Signbit => lhs.elementwise_unary(UnaryOp::Signbit),
        UFuncOperation::Exp2 => lhs.elementwise_unary(UnaryOp::Exp2),
        UFuncOperation::Fabs => lhs.elementwise_unary(UnaryOp::Fabs),
        UFuncOperation::Arccosh => lhs.elementwise_unary(UnaryOp::Arccosh),
        UFuncOperation::Arcsinh => lhs.elementwise_unary(UnaryOp::Arcsinh),
        UFuncOperation::Arctanh => lhs.elementwise_unary(UnaryOp::Arctanh),
    };

    Ok((
        out.shape().to_vec(),
        out.values().to_vec(),
        out.dtype().name().to_string(),
    ))
}

fn parse_dtype(name: &str) -> Result<DType, String> {
    DType::parse(name).ok_or_else(|| format!("unsupported dtype: {name}"))
}

fn canonical_dtype_name(name: &str) -> String {
    match name.trim().to_ascii_lowercase().as_str() {
        "f64" | "float64" => "float64".to_string(),
        "f32" | "float32" => "float32".to_string(),
        "i64" | "int64" => "int64".to_string(),
        "i32" | "int32" => "int32".to_string(),
        "u64" | "uint64" => "uint64".to_string(),
        "u32" | "uint32" => "uint32".to_string(),
        "bool" | "bool_" => "bool_".to_string(),
        other => other.to_string(),
    }
}

fn compare_arrays(
    expected_shape: &[usize],
    expected_values: &[f64],
    actual_shape: &[usize],
    actual_values: &[f64],
    abs_tol: f64,
    rel_tol: f64,
) -> (bool, f64, Option<String>) {
    if expected_shape != actual_shape {
        return (
            false,
            f64::INFINITY,
            Some(format!(
                "shape mismatch expected={expected_shape:?} actual={actual_shape:?}"
            )),
        );
    }

    if expected_values.len() != actual_values.len() {
        return (
            false,
            f64::INFINITY,
            Some(format!(
                "value length mismatch expected={} actual={}",
                expected_values.len(),
                actual_values.len()
            )),
        );
    }

    let mut max_abs_error = 0.0_f64;

    for (idx, (&expected, &actual)) in expected_values.iter().zip(actual_values).enumerate() {
        let abs_err = (expected - actual).abs();
        if abs_err > max_abs_error {
            max_abs_error = abs_err;
        }

        let threshold = abs_tol + rel_tol * expected.abs();
        if abs_err > threshold {
            return (
                false,
                max_abs_error,
                Some(format!(
                    "value mismatch at index {idx}: expected={expected} actual={actual} abs_err={abs_err} threshold={threshold}"
                )),
            );
        }
    }

    (true, max_abs_error, None)
}

#[cfg(test)]
mod tests {
    use super::{
        UFuncInputCase, UFuncOperation, UFuncOracleCapture, UFuncOracleCase,
        compare_against_oracle, execute_input_case, load_oracle_capture, write_differential_report,
    };
    use std::fs;
    use std::path::PathBuf;

    fn temp_file(name: &str) -> PathBuf {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        std::env::temp_dir().join(format!("fnp_{name}_{ts}.json"))
    }

    #[test]
    fn write_and_load_oracle_capture_roundtrip() {
        let path = temp_file("oracle_roundtrip");
        let capture = UFuncOracleCapture {
            schema_version: 1,
            oracle_source: "system".to_string(),
            generated_at_unix_ms: 1,
            cases: vec![UFuncOracleCase {
                id: "case1".to_string(),
                status: "ok".to_string(),
                error: None,
                shape: vec![1],
                values: vec![1.0],
                dtype: "float64".to_string(),
            }],
        };

        super::write_oracle_capture(&path, &capture).expect("write capture");
        let loaded = load_oracle_capture(&path).expect("load capture");
        assert_eq!(loaded.cases.len(), 1);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn compare_reports_missing_oracle_case() {
        let input_path = temp_file("input_missing");
        let oracle_path = temp_file("oracle_missing");
        let report_path = temp_file("report_missing");

        fs::write(
            &input_path,
            r#"[{"id":"x","op":"sum","lhs_shape":[1],"lhs_values":[1.0],"lhs_dtype":"f64","rhs_shape":null,"rhs_values":null,"rhs_dtype":null,"axis":null,"keepdims":false}]"#,
        )
        .expect("write input");

        fs::write(
            &oracle_path,
            r#"{"schema_version":1,"oracle_source":"system","generated_at_unix_ms":0,"cases":[]}"#,
        )
        .expect("write oracle");

        let report = compare_against_oracle(&input_path, &oracle_path, 1e-9, 1e-9)
            .expect("comparison should return report");
        assert_eq!(report.failed_cases, 1);

        write_differential_report(&report_path, &report).expect("write report");

        let _ = fs::remove_file(input_path);
        let _ = fs::remove_file(oracle_path);
        let _ = fs::remove_file(report_path);
    }

    #[test]
    fn execute_input_case_supports_negative_axis_sum() {
        let case = UFuncInputCase {
            id: "sum_axis_neg1".to_string(),
            op: UFuncOperation::Sum,
            lhs_shape: vec![2, 2, 2],
            lhs_values: (1..=8).map(f64::from).collect(),
            lhs_dtype: "f64".to_string(),
            rhs_shape: None,
            rhs_values: None,
            rhs_dtype: None,
            axis: Some(-1),
            keepdims: Some(false),
            ddof: None,
            clip_min: None,
            clip_max: None,
            third_shape: None,
            third_values: None,
            third_dtype: None,
            seed: 0,
            mode: "strict".to_string(),
            env_fingerprint: "tests".to_string(),
            artifact_refs: Vec::new(),
            reason_code: "ufunc_reduction_contract_violation".to_string(),
            expected_reason_code: "ufunc_reduction_contract_violation".to_string(),
            expected_error_contains: String::new(),
        };

        let (shape, values, dtype) = execute_input_case(&case).expect("negative axis sum");
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(values, vec![3.0, 7.0, 11.0, 15.0]);
        assert_eq!(dtype, "f64");
    }

    #[test]
    fn execute_input_case_rejects_negative_axis_out_of_bounds() {
        let case = UFuncInputCase {
            id: "sum_axis_neg3_oob".to_string(),
            op: UFuncOperation::Sum,
            lhs_shape: vec![2, 2],
            lhs_values: vec![1.0, 2.0, 3.0, 4.0],
            lhs_dtype: "f64".to_string(),
            rhs_shape: None,
            rhs_values: None,
            rhs_dtype: None,
            axis: Some(-3),
            keepdims: Some(false),
            ddof: None,
            clip_min: None,
            clip_max: None,
            third_shape: None,
            third_values: None,
            third_dtype: None,
            seed: 0,
            mode: "strict".to_string(),
            env_fingerprint: "tests".to_string(),
            artifact_refs: Vec::new(),
            reason_code: "ufunc_reduction_contract_violation".to_string(),
            expected_reason_code: "ufunc_reduction_contract_violation".to_string(),
            expected_error_contains: String::new(),
        };

        let err = execute_input_case(&case).expect_err("axis should be rejected");
        assert!(
            err.contains("axis"),
            "expected axis error substring, got: {err}"
        );
    }
}
