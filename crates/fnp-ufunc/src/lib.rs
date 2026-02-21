#![forbid(unsafe_code)]

use fnp_dtype::{DType, promote, promote_for_mean_reduction, promote_for_sum_reduction};
use fnp_ndarray::{ShapeError, broadcast_shape, element_count, fix_unknown_dimension};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UFuncRuntimeMode {
    Strict,
    Hardened,
}

impl UFuncRuntimeMode {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Hardened => "hardened",
        }
    }
}

pub const UFUNC_PACKET_REASON_CODES: [&str; 16] = [
    "ufunc_shape_contract_violation",
    "ufunc_invalid_input_length",
    "ufunc_axis_out_of_bounds",
    "ufunc_division_by_zero_observed",
    "ufunc_broadcast_selector_determinism",
    "ufunc_reduce_keepdims_contract",
    "ufunc_reduce_axis_contract",
    "ufunc_scalar_broadcast_contract",
    "ufunc_dtype_promotion_contract",
    "ufunc_signature_conflict",
    "ufunc_signature_parse_failed",
    "ufunc_fixed_signature_invalid",
    "ufunc_override_precedence_violation",
    "gufunc_loop_exception_propagated",
    "ufunc_loop_registry_invalid",
    "ufunc_policy_unknown_metadata",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Power,
    Remainder,
    Minimum,
    Maximum,
    Arctan2,
    Fmod,
    Copysign,
    Fmax,
    Fmin,
    Heaviside,
    Nextafter,
    LogicalAnd,
    LogicalOr,
    LogicalXor,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Hypot,
    Logaddexp,
    Logaddexp2,
    Ldexp,
    FloorDivide,
    FloatPower,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    LeftShift,
    RightShift,
}

impl BinaryOp {
    /// Returns `true` if this operation always produces Bool output regardless of input dtype.
    #[must_use]
    pub const fn is_bool_output(self) -> bool {
        matches!(
            self,
            Self::LogicalAnd
                | Self::LogicalOr
                | Self::LogicalXor
                | Self::Equal
                | Self::NotEqual
                | Self::Less
                | Self::LessEqual
                | Self::Greater
                | Self::GreaterEqual
        )
    }

    #[must_use]
    pub fn apply(self, lhs: f64, rhs: f64) -> f64 {
        match self {
            Self::Add => lhs + rhs,
            Self::Sub => lhs - rhs,
            Self::Mul => lhs * rhs,
            Self::Div => lhs / rhs,
            Self::Power => lhs.powf(rhs),
            Self::Remainder => {
                // NumPy remainder follows Python semantics: sign of result matches divisor
                if rhs == 0.0 {
                    f64::NAN
                } else {
                    lhs - (lhs / rhs).floor() * rhs
                }
            }
            Self::Minimum => {
                if lhs.is_nan() || rhs.is_nan() {
                    f64::NAN
                } else {
                    lhs.min(rhs)
                }
            }
            Self::Maximum => {
                if lhs.is_nan() || rhs.is_nan() {
                    f64::NAN
                } else {
                    lhs.max(rhs)
                }
            }
            Self::Arctan2 => lhs.atan2(rhs),
            Self::Fmod => {
                // C-style fmod: sign of result matches dividend
                if rhs == 0.0 { f64::NAN } else { lhs % rhs }
            }
            Self::Copysign => lhs.copysign(rhs),
            // fmax/fmin ignore NaN (return the non-NaN value)
            Self::Fmax => lhs.max(rhs),
            Self::Fmin => lhs.min(rhs),
            Self::Heaviside => {
                if lhs.is_nan() {
                    f64::NAN
                } else if lhs < 0.0 {
                    0.0
                } else if lhs == 0.0 {
                    rhs
                } else {
                    1.0
                }
            }
            Self::Nextafter => {
                if lhs.is_nan() || rhs.is_nan() {
                    f64::NAN
                } else if lhs == rhs {
                    rhs
                } else if lhs < rhs {
                    // Move toward +inf
                    if lhs == 0.0 {
                        f64::from_bits(1)
                    } else if lhs > 0.0 {
                        f64::from_bits(lhs.to_bits() + 1)
                    } else {
                        f64::from_bits(lhs.to_bits() - 1)
                    }
                } else {
                    // Move toward -inf
                    if lhs == 0.0 {
                        f64::from_bits((1_u64 << 63) | 1)
                    } else if lhs > 0.0 {
                        f64::from_bits(lhs.to_bits() - 1)
                    } else {
                        f64::from_bits(lhs.to_bits() + 1)
                    }
                }
            }
            // Logical: truthiness is x != 0.0 (NaN is truthy)
            Self::LogicalAnd => {
                if lhs != 0.0 && rhs != 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::LogicalOr => {
                if lhs != 0.0 || rhs != 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::LogicalXor => {
                if (lhs != 0.0) != (rhs != 0.0) {
                    1.0
                } else {
                    0.0
                }
            }
            // Comparisons: IEEE 754 semantics (NaN comparisons return false)
            Self::Equal => {
                if lhs == rhs {
                    1.0
                } else {
                    0.0
                }
            }
            Self::NotEqual => {
                if lhs != rhs {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Less => {
                if lhs < rhs {
                    1.0
                } else {
                    0.0
                }
            }
            Self::LessEqual => {
                if lhs <= rhs {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Greater => {
                if lhs > rhs {
                    1.0
                } else {
                    0.0
                }
            }
            Self::GreaterEqual => {
                if lhs >= rhs {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Hypot => lhs.hypot(rhs),
            Self::Logaddexp => {
                // log(exp(lhs) + exp(rhs)), numerically stable
                let max = lhs.max(rhs);
                let min = lhs.min(rhs);
                if max.is_infinite() && max.is_sign_positive() {
                    f64::INFINITY
                } else if max.is_nan() || min.is_nan() {
                    f64::NAN
                } else {
                    max + (min - max).exp().ln_1p()
                }
            }
            Self::Logaddexp2 => {
                // log2(2^lhs + 2^rhs), numerically stable
                let max = lhs.max(rhs);
                let min = lhs.min(rhs);
                if max.is_infinite() && max.is_sign_positive() {
                    f64::INFINITY
                } else if max.is_nan() || min.is_nan() {
                    f64::NAN
                } else {
                    let diff = min - max;
                    max + diff.exp2().ln_1p() / std::f64::consts::LN_2
                }
            }
            Self::Ldexp => {
                // ldexp(lhs, rhs) = lhs * 2^rhs
                lhs * (2.0_f64).powf(rhs)
            }
            Self::FloorDivide => (lhs / rhs).floor(),
            Self::FloatPower => lhs.powf(rhs),
            Self::BitwiseAnd => {
                let a = lhs as i64;
                let b = rhs as i64;
                (a & b) as f64
            }
            Self::BitwiseOr => {
                let a = lhs as i64;
                let b = rhs as i64;
                (a | b) as f64
            }
            Self::BitwiseXor => {
                let a = lhs as i64;
                let b = rhs as i64;
                (a ^ b) as f64
            }
            Self::LeftShift => {
                let a = lhs as i64;
                let b = rhs as i64;
                (a << (b & 63)) as f64
            }
            Self::RightShift => {
                let a = lhs as i64;
                let b = rhs as i64;
                (a >> (b & 63)) as f64
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
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
    Cbrt,
    Expm1,
    Log1p,
    Degrees,
    Radians,
    Rint,
    Trunc,
    Positive,
    Spacing,
    LogicalNot,
    Isnan,
    Isinf,
    Isfinite,
    Signbit,
    Exp2,
    Fabs,
    Arccosh,
    Arcsinh,
    Arctanh,
    Invert,
}

impl UnaryOp {
    /// Returns `true` if this operation always produces Bool output regardless of input dtype.
    #[must_use]
    pub const fn is_bool_output(self) -> bool {
        matches!(
            self,
            Self::LogicalNot | Self::Isnan | Self::Isinf | Self::Isfinite | Self::Signbit
        )
    }

    #[must_use]
    pub fn apply(self, x: f64) -> f64 {
        match self {
            Self::Abs => x.abs(),
            Self::Negative => -x,
            Self::Sign => {
                if x.is_nan() {
                    f64::NAN
                } else if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    x
                }
            }
            Self::Sqrt => x.sqrt(),
            Self::Square => x * x,
            Self::Exp => x.exp(),
            Self::Log => x.ln(),
            Self::Log2 => x.log2(),
            Self::Log10 => x.log10(),
            Self::Sin => x.sin(),
            Self::Cos => x.cos(),
            Self::Tan => x.tan(),
            Self::Floor => x.floor(),
            Self::Ceil => x.ceil(),
            Self::Round => x.round_ties_even(),
            Self::Reciprocal => 1.0 / x,
            Self::Sinh => x.sinh(),
            Self::Cosh => x.cosh(),
            Self::Tanh => x.tanh(),
            Self::Arcsin => x.asin(),
            Self::Arccos => x.acos(),
            Self::Arctan => x.atan(),
            Self::Cbrt => x.cbrt(),
            Self::Expm1 => x.exp_m1(),
            Self::Log1p => x.ln_1p(),
            Self::Degrees => x.to_degrees(),
            Self::Radians => x.to_radians(),
            Self::Rint => x.round_ties_even(),
            Self::Trunc => x.trunc(),
            Self::Positive => x,
            Self::Spacing => {
                if x.is_nan() || x.is_infinite() {
                    f64::NAN
                } else {
                    let abs_x = x.abs();
                    if abs_x == f64::MAX {
                        f64::MAX - f64::from_bits(f64::MAX.to_bits() - 1)
                    } else {
                        // ULP: distance to the next representable float
                        let next = f64::from_bits(abs_x.to_bits() + 1);
                        next - abs_x
                    }
                }
            }
            // Truthiness: 0.0 is false, everything else (including NaN) is true
            Self::LogicalNot => {
                if x == 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Isnan => {
                if x.is_nan() {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Isinf => {
                if x.is_infinite() {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Isfinite => {
                if x.is_finite() {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Signbit => {
                if x.is_sign_negative() {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Exp2 => x.exp2(),
            Self::Fabs => x.abs(),
            Self::Arccosh => x.acosh(),
            Self::Arcsinh => x.asinh(),
            Self::Arctanh => x.atanh(),
            Self::Invert => {
                let a = x as i64;
                (!a) as f64
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GufuncSignature {
    pub inputs: Vec<Vec<String>>,
    pub outputs: Vec<Vec<String>>,
    canonical: String,
}

impl GufuncSignature {
    #[must_use]
    pub fn canonical(&self) -> &str {
        &self.canonical
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryDispatchPlan {
    pub out_shape: Vec<usize>,
    pub out_count: usize,
    pub out_dtype: DType,
}

pub fn normalize_signature_keywords(
    sig: Option<&str>,
    signature: Option<&str>,
) -> Result<Option<String>, UFuncError> {
    let sig_trimmed = sig.map(str::trim);
    let signature_trimmed = signature.map(str::trim);

    if sig.is_some() && sig_trimmed.is_some_and(str::is_empty) {
        return Err(UFuncError::FixedSignatureInvalid {
            detail: "sig keyword must not be empty".to_string(),
        });
    }

    if signature.is_some() && signature_trimmed.is_some_and(str::is_empty) {
        return Err(UFuncError::FixedSignatureInvalid {
            detail: "signature keyword must not be empty".to_string(),
        });
    }

    match (sig_trimmed, signature_trimmed) {
        (Some(sig_raw), Some(signature_raw)) => {
            if sig_raw != signature_raw {
                return Err(UFuncError::SignatureConflict {
                    sig: sig_raw.to_string(),
                    signature: signature_raw.to_string(),
                });
            }
            Ok(Some(sig_raw.to_string()))
        }
        (Some(sig_raw), None) => Ok(Some(sig_raw.to_string())),
        (None, Some(signature_raw)) => Ok(Some(signature_raw.to_string())),
        (None, None) => Ok(None),
    }
}

pub fn parse_gufunc_signature(
    sig: Option<&str>,
    signature: Option<&str>,
) -> Result<Option<GufuncSignature>, UFuncError> {
    let Some(normalized) = normalize_signature_keywords(sig, signature)? else {
        return Ok(None);
    };

    let (inputs_raw, outputs_raw) =
        normalized
            .split_once("->")
            .ok_or_else(|| UFuncError::SignatureParse {
                detail: format!(
                    "signature '{}' must contain exactly one '->' separator",
                    normalized
                ),
            })?;

    if outputs_raw.contains("->") {
        return Err(UFuncError::SignatureParse {
            detail: format!(
                "signature '{}' must contain exactly one '->' separator",
                normalized
            ),
        });
    }

    let inputs = parse_signature_groups(inputs_raw, "input signature")?;
    let outputs = parse_signature_groups(outputs_raw, "output signature")?;
    let canonical = format!(
        "{}->{}",
        canonicalize_signature_groups(&inputs),
        canonicalize_signature_groups(&outputs)
    );
    Ok(Some(GufuncSignature {
        inputs,
        outputs,
        canonical,
    }))
}

pub fn validate_override_payload_class(payload_class: &str) -> Result<(), UFuncError> {
    let normalized = payload_class.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return Err(UFuncError::OverridePrecedenceViolation {
            detail: "override payload class must not be empty".to_string(),
        });
    }

    if matches!(
        normalized.as_str(),
        "ndarray" | "notimplemented" | "not_implemented" | "override_result"
    ) {
        Ok(())
    } else {
        Err(UFuncError::OverridePrecedenceViolation {
            detail: format!("unsupported override payload class '{payload_class}'"),
        })
    }
}

pub fn register_custom_loop(loop_name: &str) -> Result<(), UFuncError> {
    let loop_name = loop_name.trim();
    if loop_name.is_empty() {
        return Err(UFuncError::LoopRegistryInvalid {
            detail: "custom loop name must not be empty".to_string(),
        });
    }

    Err(UFuncError::LoopRegistryInvalid {
        detail: format!("custom loop registration unsupported for '{loop_name}'"),
    })
}

pub fn plan_binary_dispatch(
    lhs: &UFuncArray,
    rhs: &UFuncArray,
) -> Result<BinaryDispatchPlan, UFuncError> {
    let out_shape = broadcast_shape(lhs.shape(), rhs.shape()).map_err(UFuncError::Shape)?;
    let out_count = element_count(&out_shape).map_err(UFuncError::Shape)?;
    let out_dtype = promote(lhs.dtype(), rhs.dtype());
    Ok(BinaryDispatchPlan {
        out_shape,
        out_count,
        out_dtype,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct UFuncArray {
    shape: Vec<usize>,
    values: Vec<f64>,
    dtype: DType,
}

impl UFuncArray {
    pub fn new(shape: Vec<usize>, values: Vec<f64>, dtype: DType) -> Result<Self, UFuncError> {
        let expected = element_count(&shape).map_err(UFuncError::Shape)?;
        if values.len() != expected {
            return Err(UFuncError::InvalidInputLength {
                expected,
                actual: values.len(),
            });
        }
        Ok(Self {
            shape,
            values,
            dtype,
        })
    }

    #[must_use]
    pub fn scalar(value: f64, dtype: DType) -> Self {
        Self {
            shape: Vec::new(),
            values: vec![value],
            dtype,
        }
    }

    // ── Array creation functions ──────────────────────────────────────

    /// Create an array filled with zeros.
    pub fn zeros(shape: Vec<usize>, dtype: DType) -> Result<Self, UFuncError> {
        let count = element_count(&shape).map_err(UFuncError::Shape)?;
        Ok(Self {
            shape,
            values: vec![0.0; count],
            dtype,
        })
    }

    /// Create a zero-filled array with the same shape and dtype as another.
    pub fn zeros_like(other: &Self) -> Self {
        Self {
            shape: other.shape.clone(),
            values: vec![0.0; other.values.len()],
            dtype: other.dtype,
        }
    }

    /// Create an array filled with ones.
    pub fn ones(shape: Vec<usize>, dtype: DType) -> Result<Self, UFuncError> {
        let count = element_count(&shape).map_err(UFuncError::Shape)?;
        Ok(Self {
            shape,
            values: vec![1.0; count],
            dtype,
        })
    }

    /// Create a ones-filled array with the same shape and dtype as another.
    pub fn ones_like(other: &Self) -> Self {
        Self {
            shape: other.shape.clone(),
            values: vec![1.0; other.values.len()],
            dtype: other.dtype,
        }
    }

    /// Create an array filled with a given value.
    pub fn full(shape: Vec<usize>, fill_value: f64, dtype: DType) -> Result<Self, UFuncError> {
        let count = element_count(&shape).map_err(UFuncError::Shape)?;
        Ok(Self {
            shape,
            values: vec![fill_value; count],
            dtype,
        })
    }

    /// Create a filled array with the same shape and dtype as another.
    pub fn full_like(other: &Self, fill_value: f64) -> Self {
        Self {
            shape: other.shape.clone(),
            values: vec![fill_value; other.values.len()],
            dtype: other.dtype,
        }
    }

    /// Create an uninitialized array (filled with zeros in safe Rust).
    pub fn empty(shape: Vec<usize>, dtype: DType) -> Result<Self, UFuncError> {
        Self::zeros(shape, dtype)
    }

    /// Create an uninitialized array with the same shape/dtype as another.
    pub fn empty_like(other: &Self) -> Self {
        Self::zeros_like(other)
    }

    /// Create an array with evenly spaced values in a half-open interval.
    pub fn arange(start: f64, stop: f64, step: f64, dtype: DType) -> Result<Self, UFuncError> {
        if step == 0.0 {
            return Err(UFuncError::Msg("arange step must be non-zero".to_string()));
        }
        let n = ((stop - start) / step).ceil().max(0.0) as usize;
        let values: Vec<f64> = (0..n).map(|i| start + step * i as f64).collect();
        Ok(Self {
            shape: vec![n],
            values,
            dtype,
        })
    }

    /// Create an array with evenly spaced values over a closed interval.
    pub fn linspace(start: f64, stop: f64, num: usize, dtype: DType) -> Result<Self, UFuncError> {
        if num == 0 {
            return Ok(Self {
                shape: vec![0],
                values: Vec::new(),
                dtype,
            });
        }
        if num == 1 {
            return Ok(Self {
                shape: vec![1],
                values: vec![start],
                dtype,
            });
        }
        let step = (stop - start) / (num - 1) as f64;
        let values: Vec<f64> = (0..num).map(|i| start + step * i as f64).collect();
        Ok(Self {
            shape: vec![num],
            values,
            dtype,
        })
    }

    /// Create an array with values spaced evenly on a log scale.
    ///
    /// Mimics `np.logspace(start, stop, num, base=10.0)`.
    pub fn logspace(
        start: f64,
        stop: f64,
        num: usize,
        base: f64,
        dtype: DType,
    ) -> Result<Self, UFuncError> {
        let lin = Self::linspace(start, stop, num, dtype)?;
        let values: Vec<f64> = lin.values.iter().map(|&v| base.powf(v)).collect();
        Ok(Self {
            shape: vec![num],
            values,
            dtype,
        })
    }

    /// Create an array with values spaced evenly on a geometric (multiplicative) scale.
    ///
    /// Mimics `np.geomspace(start, stop, num)`.
    pub fn geomspace(start: f64, stop: f64, num: usize, dtype: DType) -> Result<Self, UFuncError> {
        if start == 0.0 || stop == 0.0 {
            return Err(UFuncError::Msg(
                "geomspace start and stop must be non-zero".to_string(),
            ));
        }
        if num == 0 {
            return Ok(Self {
                shape: vec![0],
                values: Vec::new(),
                dtype,
            });
        }
        if num == 1 {
            return Ok(Self {
                shape: vec![1],
                values: vec![start],
                dtype,
            });
        }
        let ratio = (stop / start).powf(1.0 / (num - 1) as f64);
        let values: Vec<f64> = (0..num).map(|i| start * ratio.powi(i as i32)).collect();
        Ok(Self {
            shape: vec![num],
            values,
            dtype,
        })
    }

    /// Create an array by applying a function to each index.
    ///
    /// Mimics `np.fromfunction(fn, shape)`. The callback receives the multi-index as `&[usize]`.
    pub fn fromfunction(
        shape: &[usize],
        dtype: DType,
        f: impl Fn(&[usize]) -> f64,
    ) -> Result<Self, UFuncError> {
        let total = element_count(shape).map_err(UFuncError::Shape)?;
        let strides = c_strides_elems(shape);
        let ndim = shape.len();
        let mut values = Vec::with_capacity(total);
        let mut idx = vec![0usize; ndim];
        for flat in 0..total {
            // Decode flat index to multi-index
            let mut rem = flat;
            for (d, dim_stride) in strides.iter().enumerate() {
                idx[d] = rem / dim_stride;
                rem %= dim_stride;
            }
            values.push(f(&idx));
        }
        Ok(Self {
            shape: shape.to_vec(),
            values,
            dtype,
        })
    }

    /// Construct an array from an iterator of values (np.fromiter).
    /// The resulting array is 1-D.
    pub fn fromiter(iter: impl IntoIterator<Item = f64>, dtype: DType) -> Self {
        let values: Vec<f64> = iter.into_iter().collect();
        let n = values.len();
        Self {
            shape: vec![n],
            values,
            dtype,
        }
    }

    /// Construct a 1-D array from a byte-like buffer of f64 values (np.frombuffer).
    /// `data` is treated as raw f64 values.
    pub fn frombuffer(data: &[f64], dtype: DType) -> Self {
        let n = data.len();
        Self {
            shape: vec![n],
            values: data.to_vec(),
            dtype,
        }
    }

    /// Check if two arrays have the same shape and elements (np.array_equal).
    pub fn array_equal(&self, other: &Self) -> bool {
        self.shape == other.shape && self.values == other.values
    }

    /// Check if two arrays are element-wise equal within broadcasting (np.array_equiv).
    /// Returns true if shapes are broadcastable and all corresponding elements match.
    pub fn array_equiv(&self, other: &Self) -> bool {
        if self.shape == other.shape {
            return self.values == other.values;
        }
        // Try broadcasting
        matches!(self.allclose(other, 0.0, 0.0), Ok(true))
    }

    /// Return an array representing the indices of a grid.
    ///
    /// Mimics `np.indices(dimensions)`. Returns shape `[ndim, *dimensions]`.
    pub fn indices(dimensions: &[usize], dtype: DType) -> Result<Self, UFuncError> {
        let ndim = dimensions.len();
        let grid_size = element_count(dimensions).map_err(UFuncError::Shape)?;
        let total = ndim * grid_size;
        let strides = c_strides_elems(dimensions);
        let mut values = Vec::with_capacity(total);

        for d in 0..ndim {
            for flat in 0..grid_size {
                let coord = (flat / strides[d]) % dimensions[d];
                values.push(coord as f64);
            }
        }

        let mut shape = Vec::with_capacity(ndim + 1);
        shape.push(ndim);
        shape.extend_from_slice(dimensions);
        Ok(Self {
            shape,
            values,
            dtype,
        })
    }

    /// Create a square identity matrix.
    ///
    /// Mimics `np.identity(n)`. Equivalent to `eye(n, None, 0, dtype)`.
    pub fn identity(n: usize, dtype: DType) -> Result<Self, UFuncError> {
        Self::eye(n, None, 0, dtype)
    }

    /// Create a lower-triangular array of ones.
    ///
    /// Mimics `np.tri(N, M, k)`.
    pub fn tri(n: usize, m: Option<usize>, k: i64, dtype: DType) -> Self {
        let cols = m.unwrap_or(n);
        let mut values = vec![0.0; n * cols];
        for r in 0..n {
            for c in 0..cols {
                if (c as i64) <= (r as i64).saturating_add(k) {
                    values[r * cols + c] = 1.0;
                }
            }
        }
        Self {
            shape: vec![n, cols],
            values,
            dtype,
        }
    }

    /// Create a diagonal matrix from a 1-D array with optional offset.
    ///
    /// Mimics `np.diagflat(v, k)`. Unlike `diag`, this always flattens the input first.
    pub fn diagflat(&self, k: i64) -> Self {
        let flat = &self.values;
        let n = flat.len();
        let abs_k = k.unsigned_abs() as usize;
        let size = n + abs_k;
        let mut values = vec![0.0; size * size];
        for (i, &val) in flat.iter().enumerate() {
            let row = if k >= 0 { i } else { i + abs_k };
            let col = if k >= 0 { i + abs_k } else { i };
            values[row * size + col] = val;
        }
        Self {
            shape: vec![size, size],
            values,
            dtype: self.dtype,
        }
    }

    /// Create a 2-D identity matrix (or offset-diagonal matrix).
    pub fn eye(n: usize, m: Option<usize>, k: i64, dtype: DType) -> Result<Self, UFuncError> {
        let cols = m.unwrap_or(n);
        let count = n * cols;
        let mut values = vec![0.0; count];
        for row in 0..n {
            let col = (row as i64).saturating_add(k);
            if col >= 0 && (col as usize) < cols {
                values[row * cols + col as usize] = 1.0;
            }
        }
        Ok(Self {
            shape: vec![n, cols],
            values,
            dtype,
        })
    }

    /// Extract a diagonal or construct a diagonal array.
    pub fn diag(&self, k: i64) -> Result<Self, UFuncError> {
        if self.shape.len() == 1 {
            // 1-D input: construct diagonal matrix
            let n = self.shape[0];
            let abs_k = k.unsigned_abs() as usize;
            let size = n + abs_k;
            let mut values = vec![0.0; size * size];
            for i in 0..n {
                let row = if k >= 0 { i } else { i + abs_k };
                let col = if k >= 0 { i + abs_k } else { i };
                values[row * size + col] = self.values[i];
            }
            Ok(Self {
                shape: vec![size, size],
                values,
                dtype: self.dtype,
            })
        } else if self.shape.len() == 2 {
            // 2-D input: extract diagonal
            let (rows, cols) = (self.shape[0], self.shape[1]);
            let start_row = if k < 0 { (-k) as usize } else { 0 };
            let start_col = if k >= 0 { k as usize } else { 0 };
            let diag_len = rows
                .saturating_sub(start_row)
                .min(cols.saturating_sub(start_col));
            let values: Vec<f64> = (0..diag_len)
                .map(|i| self.values[(start_row + i) * cols + start_col + i])
                .collect();
            Ok(Self {
                shape: vec![diag_len],
                values,
                dtype: self.dtype,
            })
        } else {
            Err(UFuncError::Msg(
                "diag requires 1-D or 2-D input".to_string(),
            ))
        }
    }

    /// Extract upper triangle of a matrix.
    pub fn triu(&self, k: i64) -> Result<Self, UFuncError> {
        if self.shape.len() != 2 {
            return Err(UFuncError::Msg("triu requires 2-D input".to_string()));
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut values = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                if c as i64 >= (r as i64).saturating_add(k) {
                    values[r * cols + c] = self.values[r * cols + c];
                }
            }
        }
        Ok(Self {
            shape: self.shape.clone(),
            values,
            dtype: self.dtype,
        })
    }

    /// Extract lower triangle of a matrix.
    pub fn tril(&self, k: i64) -> Result<Self, UFuncError> {
        if self.shape.len() != 2 {
            return Err(UFuncError::Msg("tril requires 2-D input".to_string()));
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut values = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                if c as i64 <= (r as i64).saturating_add(k) {
                    values[r * cols + c] = self.values[r * cols + c];
                }
            }
        }
        Ok(Self {
            shape: self.shape.clone(),
            values,
            dtype: self.dtype,
        })
    }

    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn elementwise_binary(&self, rhs: &Self, op: BinaryOp) -> Result<Self, UFuncError> {
        let plan = plan_binary_dispatch(self, rhs)?;
        let out_shape = plan.out_shape;
        let out_count = plan.out_count;
        let out_dtype = if op.is_bool_output() {
            DType::Bool
        } else {
            plan.out_dtype
        };

        if self.shape == rhs.shape {
            let values = self
                .values
                .iter()
                .zip(&rhs.values)
                .map(|(&lhs, &rhs)| op.apply(lhs, rhs))
                .collect::<Vec<_>>();
            return Ok(Self {
                shape: out_shape,
                values,
                dtype: out_dtype,
            });
        }

        let lhs_strides = contiguous_strides_elems(&self.shape);
        let rhs_strides = contiguous_strides_elems(&rhs.shape);
        let lhs_axis_steps =
            aligned_broadcast_axis_steps(out_shape.len(), &self.shape, &lhs_strides);
        let rhs_axis_steps =
            aligned_broadcast_axis_steps(out_shape.len(), &rhs.shape, &rhs_strides);

        let mut out_multi = vec![0usize; out_shape.len()];
        let mut lhs_flat = 0usize;
        let mut rhs_flat = 0usize;
        let mut out_values = Vec::with_capacity(out_count);

        for flat in 0..out_count {
            out_values.push(op.apply(self.values[lhs_flat], rhs.values[rhs_flat]));

            // Increment the output index as an odometer and adjust source flat
            // indices incrementally. This avoids re-unraveling and re-mapping
            // every output element.
            if flat + 1 == out_count || out_shape.is_empty() {
                continue;
            }

            for axis in (0..out_shape.len()).rev() {
                out_multi[axis] += 1;
                lhs_flat += lhs_axis_steps[axis];
                rhs_flat += rhs_axis_steps[axis];

                if out_multi[axis] < out_shape[axis] {
                    break;
                }

                out_multi[axis] = 0;
                lhs_flat -= lhs_axis_steps[axis] * out_shape[axis];
                rhs_flat -= rhs_axis_steps[axis] * out_shape[axis];
            }
        }

        Ok(Self {
            shape: out_shape,
            values: out_values,
            dtype: out_dtype,
        })
    }

    #[must_use]
    pub fn elementwise_unary(&self, op: UnaryOp) -> Self {
        let values = self.values.iter().map(|&v| op.apply(v)).collect();
        let dtype = if op.is_bool_output() {
            DType::Bool
        } else {
            self.dtype
        };
        Self {
            shape: self.shape.clone(),
            values,
            dtype,
        }
    }

    pub fn reduce_sum(&self, axis: Option<isize>, keepdims: bool) -> Result<Self, UFuncError> {
        let out_dtype = promote_for_sum_reduction(self.dtype);
        match axis {
            None => {
                let sum: f64 = self.values.iter().copied().sum();
                let shape = if keepdims {
                    vec![1; self.shape.len()]
                } else {
                    Vec::new()
                };
                Ok(Self {
                    shape,
                    values: vec![sum],
                    dtype: out_dtype,
                })
            }
            Some(axis) => {
                let axis = normalize_axis(axis, self.shape.len())?;
                let out_shape = reduced_shape(&self.shape, axis, keepdims);
                let out_count = element_count(&out_shape).map_err(UFuncError::Shape)?;
                let mut out_values = vec![0.0f64; out_count];
                reduce_sum_axis_contiguous(&self.values, &self.shape, axis, &mut out_values);

                Ok(Self {
                    shape: out_shape,
                    values: out_values,
                    dtype: out_dtype,
                })
            }
        }
    }

    pub fn reduce_prod(&self, axis: Option<isize>, keepdims: bool) -> Result<Self, UFuncError> {
        let out_dtype = promote_for_sum_reduction(self.dtype);
        match axis {
            None => {
                let prod: f64 = self.values.iter().copied().product();
                let shape = if keepdims {
                    vec![1; self.shape.len()]
                } else {
                    Vec::new()
                };
                Ok(Self {
                    shape,
                    values: vec![prod],
                    dtype: out_dtype,
                })
            }
            Some(axis) => {
                let axis = normalize_axis(axis, self.shape.len())?;
                let out_shape = reduced_shape(&self.shape, axis, keepdims);
                let out_count = element_count(&out_shape).map_err(UFuncError::Shape)?;
                let mut out_values = vec![1.0f64; out_count];
                reduce_fold_axis_contiguous(
                    &self.values,
                    &self.shape,
                    axis,
                    &mut out_values,
                    |acc, v| acc * v,
                );

                Ok(Self {
                    shape: out_shape,
                    values: out_values,
                    dtype: out_dtype,
                })
            }
        }
    }

    pub fn reduce_min(&self, axis: Option<isize>, keepdims: bool) -> Result<Self, UFuncError> {
        match axis {
            None => {
                let min = self.values.iter().copied().fold(f64::INFINITY, f64::min);
                let shape = if keepdims {
                    vec![1; self.shape.len()]
                } else {
                    Vec::new()
                };
                Ok(Self {
                    shape,
                    values: vec![min],
                    dtype: self.dtype,
                })
            }
            Some(axis) => {
                let axis = normalize_axis(axis, self.shape.len())?;
                let out_shape = reduced_shape(&self.shape, axis, keepdims);
                let out_count = element_count(&out_shape).map_err(UFuncError::Shape)?;
                let mut out_values = vec![f64::INFINITY; out_count];
                reduce_fold_axis_contiguous(
                    &self.values,
                    &self.shape,
                    axis,
                    &mut out_values,
                    f64::min,
                );

                Ok(Self {
                    shape: out_shape,
                    values: out_values,
                    dtype: self.dtype,
                })
            }
        }
    }

    pub fn reduce_max(&self, axis: Option<isize>, keepdims: bool) -> Result<Self, UFuncError> {
        match axis {
            None => {
                let max = self
                    .values
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                let shape = if keepdims {
                    vec![1; self.shape.len()]
                } else {
                    Vec::new()
                };
                Ok(Self {
                    shape,
                    values: vec![max],
                    dtype: self.dtype,
                })
            }
            Some(axis) => {
                let axis = normalize_axis(axis, self.shape.len())?;
                let out_shape = reduced_shape(&self.shape, axis, keepdims);
                let out_count = element_count(&out_shape).map_err(UFuncError::Shape)?;
                let mut out_values = vec![f64::NEG_INFINITY; out_count];
                reduce_fold_axis_contiguous(
                    &self.values,
                    &self.shape,
                    axis,
                    &mut out_values,
                    f64::max,
                );

                Ok(Self {
                    shape: out_shape,
                    values: out_values,
                    dtype: self.dtype,
                })
            }
        }
    }

    pub fn reduce_mean(&self, axis: Option<isize>, keepdims: bool) -> Result<Self, UFuncError> {
        let out_dtype = promote_for_mean_reduction(self.dtype);
        match axis {
            None => {
                let n = self.values.len() as f64;
                let sum: f64 = self.values.iter().copied().sum();
                let shape = if keepdims {
                    vec![1; self.shape.len()]
                } else {
                    Vec::new()
                };
                Ok(Self {
                    shape,
                    values: vec![sum / n],
                    dtype: out_dtype,
                })
            }
            Some(axis) => {
                let axis = normalize_axis(axis, self.shape.len())?;
                let axis_len = self.shape[axis] as f64;
                let out_shape = reduced_shape(&self.shape, axis, keepdims);
                let out_count = element_count(&out_shape).map_err(UFuncError::Shape)?;
                let mut out_values = vec![0.0f64; out_count];
                reduce_sum_axis_contiguous(&self.values, &self.shape, axis, &mut out_values);
                for v in &mut out_values {
                    *v /= axis_len;
                }

                Ok(Self {
                    shape: out_shape,
                    values: out_values,
                    dtype: out_dtype,
                })
            }
        }
    }

    pub fn cumsum(&self, axis: Option<isize>) -> Result<Self, UFuncError> {
        let out_dtype = promote_for_sum_reduction(self.dtype);
        match axis {
            None => {
                // Flatten and cumsum
                let mut acc = 0.0;
                let values: Vec<f64> = self
                    .values
                    .iter()
                    .map(|&v| {
                        acc += v;
                        acc
                    })
                    .collect();
                Ok(Self {
                    shape: vec![values.len()],
                    values,
                    dtype: out_dtype,
                })
            }
            Some(axis) => {
                let axis = normalize_axis(axis, self.shape.len())?;
                cumulate_axis(&self.values, &self.shape, axis, 0.0, |acc, v| acc + v).map(
                    |values| Self {
                        shape: self.shape.clone(),
                        values,
                        dtype: out_dtype,
                    },
                )
            }
        }
    }

    pub fn cumprod(&self, axis: Option<isize>) -> Result<Self, UFuncError> {
        let out_dtype = promote_for_sum_reduction(self.dtype);
        match axis {
            None => {
                let mut acc = 1.0;
                let values: Vec<f64> = self
                    .values
                    .iter()
                    .map(|&v| {
                        acc *= v;
                        acc
                    })
                    .collect();
                Ok(Self {
                    shape: vec![values.len()],
                    values,
                    dtype: out_dtype,
                })
            }
            Some(axis) => {
                let axis = normalize_axis(axis, self.shape.len())?;
                cumulate_axis(&self.values, &self.shape, axis, 1.0, |acc, v| acc * v).map(
                    |values| Self {
                        shape: self.shape.clone(),
                        values,
                        dtype: out_dtype,
                    },
                )
            }
        }
    }

    pub fn clip(&self, min_val: f64, max_val: f64) -> Self {
        let values = self
            .values
            .iter()
            .map(|&v| v.clamp(min_val, max_val))
            .collect();
        Self {
            shape: self.shape.clone(),
            values,
            dtype: self.dtype,
        }
    }

    /// Compute population variance along `axis` (ddof=0 by default, matching `numpy.var`).
    /// Output dtype follows `promote_for_mean_reduction` (integer types promote to F64).
    pub fn reduce_var(
        &self,
        axis: Option<isize>,
        keepdims: bool,
        ddof: usize,
    ) -> Result<Self, UFuncError> {
        let out_dtype = promote_for_mean_reduction(self.dtype);
        match axis {
            None => {
                let n = self.values.len();
                let mean = self.values.iter().copied().sum::<f64>() / n as f64;
                let var = self
                    .values
                    .iter()
                    .map(|&v| (v - mean) * (v - mean))
                    .sum::<f64>()
                    / (n.saturating_sub(ddof)) as f64;
                let shape = if keepdims {
                    vec![1; self.shape.len()]
                } else {
                    Vec::new()
                };
                Ok(Self {
                    shape,
                    values: vec![var],
                    dtype: out_dtype,
                })
            }
            Some(axis) => {
                let axis = normalize_axis(axis, self.shape.len())?;
                let axis_len = self.shape[axis];
                let out_shape = reduced_shape(&self.shape, axis, keepdims);
                let out_count = element_count(&out_shape).map_err(UFuncError::Shape)?;

                // First compute means along axis
                let mut means = vec![0.0f64; out_count];
                reduce_sum_axis_contiguous(&self.values, &self.shape, axis, &mut means);
                for m in &mut means {
                    *m /= axis_len as f64;
                }

                // Then compute sum of squared deviations
                let mut var_values = vec![0.0f64; out_count];
                reduce_var_axis_contiguous(
                    &self.values,
                    &self.shape,
                    axis,
                    &means,
                    &mut var_values,
                );
                let divisor = axis_len.saturating_sub(ddof) as f64;
                for v in &mut var_values {
                    *v /= divisor;
                }

                Ok(Self {
                    shape: out_shape,
                    values: var_values,
                    dtype: out_dtype,
                })
            }
        }
    }

    /// Compute standard deviation along `axis` (ddof=0 by default, matching `numpy.std`).
    pub fn reduce_std(
        &self,
        axis: Option<isize>,
        keepdims: bool,
        ddof: usize,
    ) -> Result<Self, UFuncError> {
        let mut var_result = self.reduce_var(axis, keepdims, ddof)?;
        for v in &mut var_result.values {
            *v = v.sqrt();
        }
        Ok(var_result)
    }

    /// Return the index of the minimum value along `axis` (matching `numpy.argmin`).
    /// When `axis` is `None`, operates on the flattened array and returns a scalar.
    /// Output dtype is always `I64`.
    pub fn reduce_argmin(&self, axis: Option<isize>) -> Result<Self, UFuncError> {
        match axis {
            None => {
                if self.values.is_empty() {
                    return Err(UFuncError::EmptyReduction { op: "argmin" });
                }
                let mut min_idx = 0usize;
                let mut min_val = self.values[0];
                for (idx, &val) in self.values.iter().enumerate().skip(1) {
                    if val.is_nan() {
                        if !min_val.is_nan() {
                            min_idx = idx;
                            min_val = val;
                        }
                        continue;
                    }
                    if !min_val.is_nan() && val < min_val {
                        min_idx = idx;
                        min_val = val;
                    }
                }
                Ok(Self {
                    shape: Vec::new(),
                    values: vec![min_idx as f64],
                    dtype: DType::I64,
                })
            }
            Some(axis) => {
                let axis = normalize_axis(axis, self.shape.len())?;
                if self.shape[axis] == 0 {
                    return Err(UFuncError::EmptyReduction { op: "argmin" });
                }
                let out_shape = reduced_shape(&self.shape, axis, false);
                let out_count = element_count(&out_shape).map_err(UFuncError::Shape)?;
                let mut out_values = vec![0.0f64; out_count];
                reduce_argfold_axis_contiguous(
                    &self.values,
                    &self.shape,
                    axis,
                    &mut out_values,
                    |cur, best| cur < best,
                );
                Ok(Self {
                    shape: out_shape,
                    values: out_values,
                    dtype: DType::I64,
                })
            }
        }
    }

    /// Return the index of the maximum value along `axis` (matching `numpy.argmax`).
    /// When `axis` is `None`, operates on the flattened array and returns a scalar.
    /// Output dtype is always `I64`.
    pub fn reduce_argmax(&self, axis: Option<isize>) -> Result<Self, UFuncError> {
        match axis {
            None => {
                if self.values.is_empty() {
                    return Err(UFuncError::EmptyReduction { op: "argmax" });
                }
                let mut max_idx = 0usize;
                let mut max_val = self.values[0];
                for (idx, &val) in self.values.iter().enumerate().skip(1) {
                    if val.is_nan() {
                        if !max_val.is_nan() {
                            max_idx = idx;
                            max_val = val;
                        }
                        continue;
                    }
                    if !max_val.is_nan() && val > max_val {
                        max_idx = idx;
                        max_val = val;
                    }
                }
                Ok(Self {
                    shape: Vec::new(),
                    values: vec![max_idx as f64],
                    dtype: DType::I64,
                })
            }
            Some(axis) => {
                let axis = normalize_axis(axis, self.shape.len())?;
                if self.shape[axis] == 0 {
                    return Err(UFuncError::EmptyReduction { op: "argmax" });
                }
                let out_shape = reduced_shape(&self.shape, axis, false);
                let out_count = element_count(&out_shape).map_err(UFuncError::Shape)?;
                let mut out_values = vec![0.0f64; out_count];
                reduce_argfold_axis_contiguous(
                    &self.values,
                    &self.shape,
                    axis,
                    &mut out_values,
                    |cur, best| cur > best,
                );
                Ok(Self {
                    shape: out_shape,
                    values: out_values,
                    dtype: DType::I64,
                })
            }
        }
    }

    /// Reshape the array to `new_shape`, supporting a single `-1` dimension for inference.
    /// Matches `numpy.reshape` semantics: total element count must be preserved.
    pub fn reshape(&self, new_shape: &[isize]) -> Result<Self, UFuncError> {
        let old_count = self.values.len();
        let resolved = fix_unknown_dimension(new_shape, old_count).map_err(UFuncError::Shape)?;
        Ok(Self {
            shape: resolved,
            values: self.values.clone(),
            dtype: self.dtype,
        })
    }

    /// Transpose (permute) axes. When `axes` is `None`, reverses all dimensions.
    /// Matches `numpy.transpose` semantics.
    pub fn transpose(&self, axes: Option<&[usize]>) -> Result<Self, UFuncError> {
        let ndim = self.shape.len();
        let perm: Vec<usize> = match axes {
            Some(ax) => {
                if ax.len() != ndim {
                    return Err(UFuncError::AxisOutOfBounds {
                        axis: ax.len() as isize,
                        ndim,
                    });
                }
                // Validate permutation: all axes present exactly once
                let mut seen = vec![false; ndim];
                for &a in ax {
                    if a >= ndim {
                        return Err(UFuncError::AxisOutOfBounds {
                            axis: a as isize,
                            ndim,
                        });
                    }
                    seen[a] = true;
                }
                if seen.iter().any(|&s| !s) {
                    return Err(UFuncError::AxisOutOfBounds { axis: -1, ndim });
                }
                ax.to_vec()
            }
            None => (0..ndim).rev().collect(),
        };

        let new_shape: Vec<usize> = perm.iter().map(|&a| self.shape[a]).collect();
        let total = self.values.len();
        let mut new_values = vec![0.0f64; total];

        // Compute C-order strides (in elements) for old and new shapes
        let old_strides = c_strides_elems(&self.shape);
        let new_strides = c_strides_elems(&new_shape);

        for (flat_new, out_value) in new_values.iter_mut().enumerate() {
            // Convert flat_new to multi-index in new shape
            let mut remainder = flat_new;
            let mut flat_old = 0usize;
            for (new_axis, &new_stride) in new_strides.iter().enumerate() {
                let idx = remainder / new_stride;
                remainder %= new_stride;
                // new_axis in new layout corresponds to perm[new_axis] in old layout
                flat_old += idx * old_strides[perm[new_axis]];
            }
            *out_value = self.values[flat_old];
        }

        Ok(Self {
            shape: new_shape,
            values: new_values,
            dtype: self.dtype,
        })
    }

    /// Return a 1-D copy of the array. Matches `numpy.ndarray.flatten()`.
    #[must_use]
    pub fn flatten(&self) -> Self {
        Self {
            shape: vec![self.values.len()],
            values: self.values.clone(),
            dtype: self.dtype,
        }
    }

    /// Remove size-1 dimensions. When `axis` is `None`, removes all size-1 dims.
    /// Matches `numpy.squeeze` semantics.
    pub fn squeeze(&self, axis: Option<isize>) -> Result<Self, UFuncError> {
        match axis {
            None => {
                let new_shape: Vec<usize> =
                    self.shape.iter().copied().filter(|&d| d != 1).collect();
                let new_shape = if new_shape.is_empty() && !self.shape.is_empty() {
                    // Scalar result — keep shape empty (0-d array)
                    Vec::new()
                } else {
                    new_shape
                };
                Ok(Self {
                    shape: new_shape,
                    values: self.values.clone(),
                    dtype: self.dtype,
                })
            }
            Some(axis) => {
                let axis = normalize_axis(axis, self.shape.len())?;
                if self.shape[axis] != 1 {
                    return Err(UFuncError::Shape(ShapeError::IncompatibleElementCount {
                        old: self.shape[axis],
                        new: 1,
                    }));
                }
                let mut new_shape = self.shape.clone();
                new_shape.remove(axis);
                Ok(Self {
                    shape: new_shape,
                    values: self.values.clone(),
                    dtype: self.dtype,
                })
            }
        }
    }

    /// Insert a size-1 dimension at position `axis`. Matches `numpy.expand_dims`.
    pub fn expand_dims(&self, axis: isize) -> Result<Self, UFuncError> {
        let ndim = self.shape.len() + 1; // new ndim after insertion
        let axis = if axis < 0 {
            let normalized = ndim as isize + axis;
            if normalized < 0 {
                return Err(UFuncError::AxisOutOfBounds { axis, ndim });
            }
            normalized as usize
        } else {
            let a = axis as usize;
            if a > self.shape.len() {
                return Err(UFuncError::AxisOutOfBounds { axis, ndim });
            }
            a
        };
        let mut new_shape = self.shape.clone();
        new_shape.insert(axis, 1);
        Ok(Self {
            shape: new_shape,
            values: self.values.clone(),
            dtype: self.dtype,
        })
    }

    /// Swap two axes of an array.
    ///
    /// Mimics `np.swapaxes(a, axis1, axis2)`.
    pub fn swapaxes(&self, axis1: isize, axis2: isize) -> Result<Self, UFuncError> {
        let ndim = self.shape.len();
        let a1 = normalize_axis(axis1, ndim)?;
        let a2 = normalize_axis(axis2, ndim)?;
        let mut perm: Vec<usize> = (0..ndim).collect();
        perm.swap(a1, a2);
        self.transpose(Some(&perm))
    }

    /// Move an axis to a new position.
    ///
    /// Mimics `np.moveaxis(a, source, destination)`.
    pub fn moveaxis(&self, source: isize, destination: isize) -> Result<Self, UFuncError> {
        let ndim = self.shape.len();
        let src = normalize_axis(source, ndim)?;
        let dst = normalize_axis(destination, ndim)?;
        let mut perm: Vec<usize> = (0..ndim).collect();
        perm.remove(src);
        perm.insert(dst, src);
        self.transpose(Some(&perm))
    }

    /// Ensure the array is at least 1-D.
    ///
    /// Mimics `np.atleast_1d(a)`.
    pub fn atleast_1d(&self) -> Self {
        if self.shape.is_empty() {
            Self {
                shape: vec![1],
                values: self.values.clone(),
                dtype: self.dtype,
            }
        } else {
            self.clone()
        }
    }

    /// Ensure the array is at least 2-D.
    ///
    /// Mimics `np.atleast_2d(a)`.
    pub fn atleast_2d(&self) -> Self {
        match self.shape.len() {
            0 => Self {
                shape: vec![1, 1],
                values: self.values.clone(),
                dtype: self.dtype,
            },
            1 => Self {
                shape: vec![1, self.shape[0]],
                values: self.values.clone(),
                dtype: self.dtype,
            },
            _ => self.clone(),
        }
    }

    /// Ensure the array is at least 3-D.
    ///
    /// Mimics `np.atleast_3d(a)`.
    pub fn atleast_3d(&self) -> Self {
        match self.shape.len() {
            0 => Self {
                shape: vec![1, 1, 1],
                values: self.values.clone(),
                dtype: self.dtype,
            },
            1 => Self {
                shape: vec![1, self.shape[0], 1],
                values: self.values.clone(),
                dtype: self.dtype,
            },
            2 => {
                let mut shape = self.shape.clone();
                shape.push(1);
                Self {
                    shape,
                    values: self.values.clone(),
                    dtype: self.dtype,
                }
            }
            _ => self.clone(),
        }
    }

    /// Stack arrays along the third axis (depth).
    ///
    /// Mimics `np.dstack(arrays)`.
    pub fn dstack(arrays: &[Self]) -> Result<Self, UFuncError> {
        if arrays.is_empty() {
            return Err(UFuncError::Msg(
                "dstack requires at least one array".to_string(),
            ));
        }
        // Promote all to at least 3-D, then concatenate along axis 2
        let promoted: Vec<Self> = arrays.iter().map(|a| a.atleast_3d()).collect();
        let refs: Vec<&Self> = promoted.iter().collect();
        Self::concatenate(&refs, 2)
    }

    /// Delete elements along an axis.
    ///
    /// Mimics `np.delete(arr, indices, axis)`.
    pub fn delete(&self, indices: &[usize], axis: Option<isize>) -> Result<Self, UFuncError> {
        match axis {
            None => {
                // Flatten, then delete
                let n = self.values.len();
                for &idx in indices {
                    if idx >= n {
                        return Err(UFuncError::Msg(format!(
                            "delete: index {idx} out of bounds for size {n}"
                        )));
                    }
                }
                let mask: std::collections::HashSet<usize> = indices.iter().copied().collect();
                let values: Vec<f64> = self
                    .values
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !mask.contains(i))
                    .map(|(_, &v)| v)
                    .collect();
                let n = values.len();
                Ok(Self {
                    shape: vec![n],
                    values,
                    dtype: self.dtype,
                })
            }
            Some(ax) => {
                let ax = normalize_axis(ax, self.shape.len())?;
                let axis_len = self.shape[ax];
                for &idx in indices {
                    if idx >= axis_len {
                        return Err(UFuncError::Msg(format!(
                            "delete: index {idx} out of bounds for axis {ax} with size {axis_len}"
                        )));
                    }
                }
                let mask: std::collections::HashSet<usize> = indices.iter().copied().collect();
                let new_axis_len = axis_len - mask.len();
                let strides = c_strides_elems(&self.shape);
                let total = self.values.len();

                let mut new_shape = self.shape.clone();
                new_shape[ax] = new_axis_len;
                let new_total = element_count(&new_shape).map_err(UFuncError::Shape)?;
                let mut values = Vec::with_capacity(new_total);

                for flat in 0..total {
                    let coord_ax = (flat / strides[ax]) % axis_len;
                    if !mask.contains(&coord_ax) {
                        values.push(self.values[flat]);
                    }
                }
                Ok(Self {
                    shape: new_shape,
                    values,
                    dtype: self.dtype,
                })
            }
        }
    }

    /// Insert values along an axis.
    ///
    /// Mimics `np.insert(arr, index, values, axis)`.
    /// `index` is the position before which to insert. `insert_values` are the values to insert.
    pub fn insert(
        &self,
        index: usize,
        insert_values: &Self,
        axis: Option<isize>,
    ) -> Result<Self, UFuncError> {
        if insert_values.values.is_empty() {
            return Err(UFuncError::Msg(
                "insert: insert_values must not be empty".to_string(),
            ));
        }
        match axis {
            None => {
                let mut values = self.values.clone();
                if index > values.len() {
                    return Err(UFuncError::Msg(format!(
                        "insert: index {index} out of bounds for size {}",
                        values.len()
                    )));
                }
                let idx = index;
                for (i, &v) in insert_values.values.iter().enumerate() {
                    values.insert(idx + i, v);
                }
                let n = values.len();
                Ok(Self {
                    shape: vec![n],
                    values,
                    dtype: self.dtype,
                })
            }
            Some(ax) => {
                let ax = normalize_axis(ax, self.shape.len())?;
                let axis_len = self.shape[ax];
                if index > axis_len {
                    return Err(UFuncError::Msg(format!(
                        "insert: index {index} out of bounds for axis {ax} with size {axis_len}"
                    )));
                }
                // inner = product of dims after axis
                let inner: usize = self.shape[ax + 1..].iter().copied().product();
                let outer: usize = self.shape[..ax].iter().copied().product();

                let mut new_shape = self.shape.clone();
                new_shape[ax] = axis_len + 1;
                let mut values = Vec::with_capacity(outer * (axis_len + 1) * inner);

                for o in 0..outer {
                    for k in 0..=axis_len {
                        if k == index {
                            // Insert values for this slice
                            for i in 0..inner {
                                let insert_idx = (o * inner + i)
                                    .min(insert_values.values.len().saturating_sub(1));
                                values.push(insert_values.values[insert_idx]);
                            }
                        }
                        if k < axis_len {
                            let src_base = o * axis_len * inner + k * inner;
                            values.extend_from_slice(&self.values[src_base..src_base + inner]);
                        }
                    }
                }
                Ok(Self {
                    shape: new_shape,
                    values,
                    dtype: self.dtype,
                })
            }
        }
    }

    /// Append values to the end of an array.
    ///
    /// Mimics `np.append(arr, values, axis)`.
    pub fn append(&self, other: &Self, axis: Option<isize>) -> Result<Self, UFuncError> {
        match axis {
            None => {
                let mut values = self.values.clone();
                values.extend_from_slice(&other.values);
                let n = values.len();
                Ok(Self {
                    shape: vec![n],
                    values,
                    dtype: self.dtype,
                })
            }
            Some(ax) => Self::concatenate(&[self, other], ax),
        }
    }

    /// Rotate an array by 90 degrees in the plane specified by axes.
    ///
    /// Mimics `np.rot90(m, k=1)`. Operates on the first two axes.
    pub fn rot90(&self, k: i32) -> Result<Self, UFuncError> {
        if self.shape.len() < 2 {
            return Err(UFuncError::Msg(
                "rot90 requires at least 2-D input".to_string(),
            ));
        }
        let k = k.rem_euclid(4);
        match k {
            0 => Ok(self.clone()),
            1 => {
                // rot90(m, 1) = flip(swapaxes(m, 0, 1), axis=0)
                self.swapaxes(0, 1).and_then(|t| t.flip(Some(0)))
            }
            2 => {
                // rot90(m, 2) = flip(flip(m, 0), 1)
                self.flip(Some(0)).and_then(|f| f.flip(Some(1)))
            }
            3 => {
                // rot90(m, 3) = swapaxes(flip(m, 0), 0, 1)
                self.flip(Some(0)).and_then(|f| f.swapaxes(0, 1))
            }
            _ => unreachable!(),
        }
    }

    /// Stack 1-D arrays as columns into a 2-D array.
    ///
    /// Mimics `np.column_stack(arrays)`.
    pub fn column_stack(arrays: &[Self]) -> Result<Self, UFuncError> {
        if arrays.is_empty() {
            return Err(UFuncError::Msg(
                "column_stack requires at least one array".to_string(),
            ));
        }
        // Promote 1-D to 2-D column vectors, then hstack
        let promoted: Vec<Self> = arrays
            .iter()
            .map(|a| {
                if a.shape.len() == 1 {
                    Self {
                        shape: vec![a.shape[0], 1],
                        values: a.values.clone(),
                        dtype: a.dtype,
                    }
                } else {
                    a.clone()
                }
            })
            .collect();
        let refs: Vec<&Self> = promoted.iter().collect();
        Self::concatenate(&refs, 1)
    }

    /// Split an array into multiple sub-arrays along axis 0 (vertically).
    ///
    /// Mimics `np.vsplit(a, sections)`.
    pub fn vsplit(&self, sections: usize) -> Result<Vec<Self>, UFuncError> {
        self.split(sections, 0)
    }

    /// Split an array into multiple sub-arrays along axis 1 (horizontally).
    ///
    /// Mimics `np.hsplit(a, sections)`.
    pub fn hsplit(&self, sections: usize) -> Result<Vec<Self>, UFuncError> {
        self.split(sections, 1)
    }

    /// Split an array into multiple sub-arrays along axis 2 (depth).
    ///
    /// Mimics `np.dsplit(a, sections)`.
    pub fn dsplit(&self, sections: usize) -> Result<Vec<Self>, UFuncError> {
        self.split(sections, 2)
    }

    /// Broadcast arrays against each other.
    ///
    /// Mimics `np.broadcast_arrays(*args)`. Returns arrays with shapes broadcast to a common shape.
    pub fn broadcast_arrays(arrays: &[&Self]) -> Result<Vec<Self>, UFuncError> {
        if arrays.is_empty() {
            return Ok(Vec::new());
        }
        // Compute common broadcast shape
        let mut shape = arrays[0].shape.clone();
        for arr in &arrays[1..] {
            shape = broadcast_shape(&shape, &arr.shape).map_err(UFuncError::Shape)?;
        }
        // Broadcast each array to the common shape
        let out_count = element_count(&shape).map_err(UFuncError::Shape)?;
        let ndim = shape.len();
        let mut result = Vec::with_capacity(arrays.len());
        for arr in arrays {
            let arr_ndim = arr.shape.len();
            let arr_strides = contiguous_strides_elems(&arr.shape);
            let steps = aligned_broadcast_axis_steps(ndim, &arr.shape, &arr_strides);
            let out_strides = c_strides_elems(&shape);
            let mut values = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let mut src_idx = 0usize;
                for d in 0..ndim {
                    let coord = (flat / out_strides[d]) % shape[d];
                    let offset = ndim - arr_ndim;
                    if d >= offset && arr.shape[d - offset] > 1 {
                        src_idx += coord * steps[d];
                    }
                }
                values.push(arr.values[src_idx]);
            }
            result.push(Self {
                shape: shape.clone(),
                values,
                dtype: arr.dtype,
            });
        }
        Ok(result)
    }

    /// Trim leading and trailing zeros from a 1-D array.
    ///
    /// Mimics `np.trim_zeros(filt)`.
    pub fn trim_zeros(&self) -> Self {
        let n = self.values.len();
        let start = self.values.iter().position(|&v| v != 0.0).unwrap_or(n);
        let end = self
            .values
            .iter()
            .rposition(|&v| v != 0.0)
            .map_or(start, |p| p + 1);
        let values = self.values[start..end].to_vec();
        let len = values.len();
        Self {
            shape: vec![len],
            values,
            dtype: self.dtype,
        }
    }

    /// Return a new array with the given shape, repeating as necessary.
    ///
    /// Mimics `np.resize(a, new_shape)`.
    pub fn resize(&self, new_shape: &[usize]) -> Result<Self, UFuncError> {
        let new_count = element_count(new_shape).map_err(UFuncError::Shape)?;
        if self.values.is_empty() {
            return Ok(Self {
                shape: new_shape.to_vec(),
                values: vec![0.0; new_count],
                dtype: self.dtype,
            });
        }
        let values: Vec<f64> = (0..new_count)
            .map(|i| self.values[i % self.values.len()])
            .collect();
        Ok(Self {
            shape: new_shape.to_vec(),
            values,
            dtype: self.dtype,
        })
    }

    /// Conditional element selection: `numpy.where(condition, x, y)`.
    /// Broadcasts condition, x, and y to a common shape. Where condition is nonzero,
    /// selects from `x`; otherwise selects from `y`. Output dtype is `promote(x.dtype, y.dtype)`.
    pub fn where_select(condition: &Self, x: &Self, y: &Self) -> Result<Self, UFuncError> {
        // Broadcast all three to a common shape
        let shape_cx = broadcast_shape(&condition.shape, &x.shape).map_err(UFuncError::Shape)?;
        let out_shape = broadcast_shape(&shape_cx, &y.shape).map_err(UFuncError::Shape)?;
        let out_count = element_count(&out_shape).map_err(UFuncError::Shape)?;
        let out_dtype = promote(x.dtype, y.dtype);

        let cond_strides = contiguous_strides_elems(&condition.shape);
        let x_strides = contiguous_strides_elems(&x.shape);
        let y_strides = contiguous_strides_elems(&y.shape);
        let cond_steps =
            aligned_broadcast_axis_steps(out_shape.len(), &condition.shape, &cond_strides);
        let x_steps = aligned_broadcast_axis_steps(out_shape.len(), &x.shape, &x_strides);
        let y_steps = aligned_broadcast_axis_steps(out_shape.len(), &y.shape, &y_strides);

        let ndim = out_shape.len();
        let mut indices = vec![0usize; ndim];
        let mut cond_flat = 0usize;
        let mut x_flat = 0usize;
        let mut y_flat = 0usize;
        let mut values = Vec::with_capacity(out_count);

        for flat in 0..out_count {
            if condition.values[cond_flat] != 0.0 {
                values.push(x.values[x_flat]);
            } else {
                values.push(y.values[y_flat]);
            }

            if flat + 1 == out_count || ndim == 0 {
                continue;
            }

            // Advance odometer
            for dim in (0..ndim).rev() {
                indices[dim] += 1;
                cond_flat += cond_steps[dim];
                x_flat += x_steps[dim];
                y_flat += y_steps[dim];
                if indices[dim] < out_shape[dim] {
                    break;
                }
                indices[dim] = 0;
                cond_flat -= cond_steps[dim] * out_shape[dim];
                x_flat -= x_steps[dim] * out_shape[dim];
                y_flat -= y_steps[dim] * out_shape[dim];
            }
        }

        Ok(Self {
            shape: out_shape,
            values,
            dtype: out_dtype,
        })
    }

    /// Sort along `axis`. When `axis` is `None`, flatten and sort.
    /// Returns a new sorted array (non-mutating, like `numpy.sort`).
    pub fn sort(&self, axis: Option<isize>) -> Result<Self, UFuncError> {
        match axis {
            None => {
                let mut values = self.values.clone();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                Ok(Self {
                    shape: vec![values.len()],
                    values,
                    dtype: self.dtype,
                })
            }
            Some(axis) => {
                let axis = normalize_axis(axis, self.shape.len())?;
                let axis_len = self.shape[axis];
                if axis_len <= 1 {
                    return Ok(self.clone());
                }

                let inner: usize = self.shape[axis + 1..].iter().copied().product();
                let outer: usize = self.shape[..axis].iter().copied().product();
                let mut values = self.values.clone();

                let mut lane = vec![0.0f64; axis_len];
                for outer_idx in 0..outer {
                    let base = outer_idx * axis_len * inner;
                    for inner_idx in 0..inner {
                        // Extract lane
                        for k in 0..axis_len {
                            lane[k] = values[base + k * inner + inner_idx];
                        }
                        lane.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        // Write back
                        for k in 0..axis_len {
                            values[base + k * inner + inner_idx] = lane[k];
                        }
                    }
                }

                Ok(Self {
                    shape: self.shape.clone(),
                    values,
                    dtype: self.dtype,
                })
            }
        }
    }

    /// Return indices that would sort the array along `axis`.
    /// When `axis` is `None`, operates on the flattened array.
    pub fn argsort(&self, axis: Option<isize>) -> Result<Self, UFuncError> {
        match axis {
            None => {
                let mut indices: Vec<usize> = (0..self.values.len()).collect();
                indices.sort_by(|&a, &b| {
                    self.values[a]
                        .partial_cmp(&self.values[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                Ok(Self {
                    shape: vec![self.values.len()],
                    values: indices.iter().map(|&i| i as f64).collect(),
                    dtype: DType::I64,
                })
            }
            Some(axis) => {
                let axis = normalize_axis(axis, self.shape.len())?;
                let axis_len = self.shape[axis];
                let inner: usize = self.shape[axis + 1..].iter().copied().product();
                let outer: usize = self.shape[..axis].iter().copied().product();
                let mut out_values = vec![0.0f64; self.values.len()];

                let mut idx_lane: Vec<usize> = (0..axis_len).collect();
                for outer_idx in 0..outer {
                    let base = outer_idx * axis_len * inner;
                    for inner_idx in 0..inner {
                        idx_lane.iter_mut().enumerate().for_each(|(i, v)| *v = i);
                        idx_lane.sort_by(|&a, &b| {
                            let va = self.values[base + a * inner + inner_idx];
                            let vb = self.values[base + b * inner + inner_idx];
                            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        for k in 0..axis_len {
                            out_values[base + k * inner + inner_idx] = idx_lane[k] as f64;
                        }
                    }
                }

                Ok(Self {
                    shape: self.shape.clone(),
                    values: out_values,
                    dtype: DType::I64,
                })
            }
        }
    }

    /// Find insertion indices that preserve sort order (left side), like
    /// `numpy.searchsorted(a, v, side="left")`.
    ///
    /// Contract: `self` must be one-dimensional; `values` may be scalar or any shape.
    /// The output shape matches `values.shape` and dtype is `I64`.
    pub fn searchsorted(&self, values: &Self) -> Result<Self, UFuncError> {
        if self.shape.len() != 1 {
            return Err(UFuncError::Shape(ShapeError::RankMismatch {
                expected: 1,
                actual: self.shape.len(),
            }));
        }

        let sorted = self.values.as_slice();
        let out_values = values
            .values
            .iter()
            .map(|needle| {
                let idx = sorted.partition_point(|candidate| {
                    candidate
                        .partial_cmp(needle)
                        .is_some_and(|ord| ord == std::cmp::Ordering::Less)
                });
                idx as f64
            })
            .collect();

        Ok(Self {
            shape: values.shape.clone(),
            values: out_values,
            dtype: DType::I64,
        })
    }

    /// Partial sort: rearranges elements so that the element at `kth` position
    /// is in its final sorted position. Elements before kth are all smaller,
    /// elements after are all larger (but not necessarily sorted).
    ///
    /// Mimics `np.partition(a, kth)`. Only 1-D supported.
    pub fn partition(&self, kth: usize) -> Result<Self, UFuncError> {
        if self.shape.len() != 1 {
            return Err(UFuncError::Msg(
                "partition: only 1-D arrays supported".to_string(),
            ));
        }
        let n = self.values.len();
        if kth >= n {
            return Err(UFuncError::Msg("partition: kth out of bounds".to_string()));
        }
        let mut values = self.values.clone();
        values.select_nth_unstable_by(kth, |a, b| a.total_cmp(b));
        Ok(Self {
            shape: self.shape.clone(),
            values,
            dtype: self.dtype,
        })
    }

    /// Return indices that would partition the array.
    ///
    /// Mimics `np.argpartition(a, kth)`. Only 1-D supported.
    pub fn argpartition(&self, kth: usize) -> Result<Self, UFuncError> {
        if self.shape.len() != 1 {
            return Err(UFuncError::Msg(
                "argpartition: only 1-D arrays supported".to_string(),
            ));
        }
        let n = self.values.len();
        if kth >= n {
            return Err(UFuncError::Msg(
                "argpartition: kth out of bounds".to_string(),
            ));
        }
        let mut indices: Vec<usize> = (0..n).collect();
        indices.select_nth_unstable_by(kth, |&a, &b| self.values[a].total_cmp(&self.values[b]));
        let values: Vec<f64> = indices.iter().map(|&i| i as f64).collect();
        Ok(Self {
            shape: self.shape.clone(),
            values,
            dtype: DType::I64,
        })
    }

    /// Sort by multiple keys (last key is primary).
    ///
    /// Mimics `np.lexsort(keys)`. Each key must be 1-D with the same length.
    /// Returns sorted indices. Keys are sorted from last to first (last key is primary).
    pub fn lexsort(keys: &[&Self]) -> Result<Self, UFuncError> {
        if keys.is_empty() {
            return Err(UFuncError::Msg(
                "lexsort: at least one key required".to_string(),
            ));
        }
        let n = keys[0].values.len();
        for key in keys {
            if key.shape.len() != 1 || key.values.len() != n {
                return Err(UFuncError::Msg(
                    "lexsort: all keys must be 1-D with same length".to_string(),
                ));
            }
        }
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            // Compare from last key to first (last key is primary)
            for key in keys.iter().rev() {
                let ord = key.values[a].total_cmp(&key.values[b]);
                if ord != std::cmp::Ordering::Equal {
                    return ord;
                }
            }
            std::cmp::Ordering::Equal
        });
        let values: Vec<f64> = indices.iter().map(|&i| i as f64).collect();
        Ok(Self {
            shape: vec![n],
            values,
            dtype: DType::I64,
        })
    }

    /// Concatenate arrays along an existing axis.
    pub fn concatenate(arrays: &[&Self], axis: isize) -> Result<Self, UFuncError> {
        if arrays.is_empty() {
            return Err(UFuncError::InvalidInputLength {
                expected: 1,
                actual: 0,
            });
        }

        let first = arrays[0];
        let ndim = first.shape.len();
        let axis = normalize_axis(axis, ndim)?;

        // Validate all arrays have same ndim and matching non-axis dims
        for arr in &arrays[1..] {
            if arr.shape.len() != ndim {
                return Err(UFuncError::Shape(ShapeError::RankMismatch {
                    expected: ndim,
                    actual: arr.shape.len(),
                }));
            }
            for (d, (&a, &b)) in first.shape.iter().zip(arr.shape.iter()).enumerate() {
                if d != axis && a != b {
                    return Err(UFuncError::Shape(ShapeError::IncompatibleBroadcast {
                        lhs: first.shape.clone(),
                        rhs: arr.shape.clone(),
                    }));
                }
            }
        }

        // Compute output shape
        let concat_dim: usize = arrays.iter().map(|a| a.shape[axis]).sum();
        let mut out_shape = first.shape.clone();
        out_shape[axis] = concat_dim;
        let out_count = element_count(&out_shape).map_err(UFuncError::Shape)?;

        // Build output values
        let inner: usize = first.shape[axis + 1..].iter().copied().product();
        let outer: usize = first.shape[..axis].iter().copied().product();
        let mut out_values = vec![0.0f64; out_count];
        let out_inner = inner;

        for outer_idx in 0..outer {
            let mut write_offset = outer_idx * concat_dim * out_inner;
            for &arr in arrays {
                let arr_axis_len = arr.shape[axis];
                let read_base = outer_idx * arr_axis_len * inner;
                for k in 0..arr_axis_len {
                    for i in 0..inner {
                        out_values[write_offset + k * out_inner + i] =
                            arr.values[read_base + k * inner + i];
                    }
                }
                write_offset += arr_axis_len * out_inner;
            }
        }

        let out_dtype = arrays
            .iter()
            .skip(1)
            .fold(first.dtype, |acc, arr| promote(acc, arr.dtype));

        Ok(Self {
            shape: out_shape,
            values: out_values,
            dtype: out_dtype,
        })
    }

    /// Stack arrays along a new axis.
    pub fn stack(arrays: &[&Self], axis: isize) -> Result<Self, UFuncError> {
        if arrays.is_empty() {
            return Err(UFuncError::InvalidInputLength {
                expected: 1,
                actual: 0,
            });
        }

        let first = arrays[0];
        let result_ndim = first.shape.len() + 1;
        let axis = normalize_axis(axis, result_ndim)?;
        let axis_isize = isize::try_from(axis).map_err(|_| UFuncError::AxisOutOfBounds {
            axis: -1,
            ndim: result_ndim,
        })?;

        for arr in &arrays[1..] {
            if arr.shape.len() != first.shape.len() {
                return Err(UFuncError::Shape(ShapeError::RankMismatch {
                    expected: first.shape.len(),
                    actual: arr.shape.len(),
                }));
            }
            if arr.shape != first.shape {
                return Err(UFuncError::Shape(ShapeError::IncompatibleBroadcast {
                    lhs: first.shape.clone(),
                    rhs: arr.shape.clone(),
                }));
            }
        }

        let expanded = arrays
            .iter()
            .map(|arr| arr.expand_dims(axis_isize))
            .collect::<Result<Vec<_>, _>>()?;
        let expanded_refs: Vec<&Self> = expanded.iter().collect();
        Self::concatenate(&expanded_refs, axis_isize)
    }

    // ── Additional array manipulation operations ─────────────────────

    /// Split an array into sub-arrays along an axis.
    pub fn split(&self, sections: usize, axis: isize) -> Result<Vec<Self>, UFuncError> {
        if self.shape.is_empty() {
            return Err(UFuncError::Msg("cannot split scalar array".to_string()));
        }
        let axis = normalize_axis(axis, self.shape.len())?;
        let axis_len = self.shape[axis];
        if sections == 0 || !axis_len.is_multiple_of(sections) {
            return Err(UFuncError::Msg(format!(
                "array split does not result in an equal division: {axis_len} into {sections}"
            )));
        }
        let chunk = axis_len / sections;
        let inner: usize = self.shape[axis + 1..].iter().copied().product();
        let outer: usize = self.shape[..axis].iter().copied().product();

        let mut result = Vec::with_capacity(sections);
        for s in 0..sections {
            let mut sub_shape = self.shape.clone();
            sub_shape[axis] = chunk;
            let count = element_count(&sub_shape).map_err(UFuncError::Shape)?;
            let mut values = vec![0.0f64; count];
            for o in 0..outer {
                for k in 0..chunk {
                    let src_k = s * chunk + k;
                    for i in 0..inner {
                        values[o * chunk * inner + k * inner + i] =
                            self.values[o * axis_len * inner + src_k * inner + i];
                    }
                }
            }
            result.push(Self {
                shape: sub_shape,
                values,
                dtype: self.dtype,
            });
        }
        Ok(result)
    }

    /// Construct an array by repeating input the given number of times along each axis.
    pub fn tile(&self, reps: &[usize]) -> Result<Self, UFuncError> {
        if reps.is_empty() {
            return Ok(self.clone());
        }
        // Pad shape or reps to match lengths
        let ndim = self.shape.len().max(reps.len());
        let mut padded_shape = vec![1usize; ndim];
        let offset = ndim - self.shape.len();
        for (i, &s) in self.shape.iter().enumerate() {
            padded_shape[offset + i] = s;
        }
        let mut padded_reps = vec![1usize; ndim];
        let rep_offset = ndim - reps.len();
        for (i, &r) in reps.iter().enumerate() {
            padded_reps[rep_offset + i] = r;
        }

        let out_shape: Vec<usize> = padded_shape
            .iter()
            .zip(&padded_reps)
            .map(|(&s, &r)| s * r)
            .collect();
        let out_count = element_count(&out_shape).map_err(UFuncError::Shape)?;
        let src_count = element_count(&padded_shape).map_err(UFuncError::Shape)?;

        // Reshape source to padded shape
        let src_values = if self.values.len() == src_count {
            &self.values[..]
        } else {
            return Err(UFuncError::Msg("tile: internal shape mismatch".to_string()));
        };

        let src_strides = c_strides_elems(&padded_shape);
        let out_strides = c_strides_elems(&out_shape);
        let mut out_values = vec![0.0f64; out_count];

        for (flat, out_val) in out_values.iter_mut().enumerate() {
            let mut src_flat = 0;
            let mut remainder = flat;
            for d in 0..ndim {
                let out_idx = remainder / out_strides[d];
                remainder %= out_strides[d];
                let src_idx = out_idx % padded_shape[d];
                src_flat += src_idx * src_strides[d];
            }
            *out_val = src_values[src_flat];
        }
        Ok(Self {
            shape: out_shape,
            values: out_values,
            dtype: self.dtype,
        })
    }

    /// Repeat elements of an array along an axis or flattened.
    pub fn repeat(&self, repeats: usize, axis: Option<isize>) -> Result<Self, UFuncError> {
        if repeats == 0 {
            return match axis {
                None => Ok(Self {
                    shape: vec![0],
                    values: Vec::new(),
                    dtype: self.dtype,
                }),
                Some(ax) => {
                    let ax = normalize_axis(ax, self.shape.len())?;
                    let mut out_shape = self.shape.clone();
                    out_shape[ax] = 0;
                    Ok(Self {
                        shape: out_shape,
                        values: Vec::new(),
                        dtype: self.dtype,
                    })
                }
            };
        }
        match axis {
            None => {
                let values: Vec<f64> = self
                    .values
                    .iter()
                    .flat_map(|&v| std::iter::repeat_n(v, repeats))
                    .collect();
                let n = values.len();
                Ok(Self {
                    shape: vec![n],
                    values,
                    dtype: self.dtype,
                })
            }
            Some(ax) => {
                let ax = normalize_axis(ax, self.shape.len())?;
                let inner: usize = self.shape[ax + 1..].iter().copied().product();
                let outer: usize = self.shape[..ax].iter().copied().product();
                let axis_len = self.shape[ax];
                let new_axis_len = axis_len * repeats;

                let mut out_shape = self.shape.clone();
                out_shape[ax] = new_axis_len;
                let out_count = element_count(&out_shape).map_err(UFuncError::Shape)?;
                let mut values = vec![0.0f64; out_count];

                for o in 0..outer {
                    for k in 0..axis_len {
                        for r in 0..repeats {
                            for i in 0..inner {
                                values[o * new_axis_len * inner + (k * repeats + r) * inner + i] =
                                    self.values[o * axis_len * inner + k * inner + i];
                            }
                        }
                    }
                }
                Ok(Self {
                    shape: out_shape,
                    values,
                    dtype: self.dtype,
                })
            }
        }
    }

    /// Roll array elements along an axis or flattened.
    pub fn roll(&self, shift: isize, axis: Option<isize>) -> Result<Self, UFuncError> {
        match axis {
            None => {
                let n = self.values.len();
                if n == 0 {
                    return Ok(self.clone());
                }
                let s = ((shift % n as isize) + n as isize) as usize % n;
                let mut values = vec![0.0f64; n];
                for (i, &v) in self.values.iter().enumerate() {
                    values[(i + s) % n] = v;
                }
                Ok(Self {
                    shape: self.shape.clone(),
                    values,
                    dtype: self.dtype,
                })
            }
            Some(ax) => {
                let ax = normalize_axis(ax, self.shape.len())?;
                let inner: usize = self.shape[ax + 1..].iter().copied().product();
                let outer: usize = self.shape[..ax].iter().copied().product();
                let axis_len = self.shape[ax];
                if axis_len == 0 {
                    return Ok(self.clone());
                }
                let s = ((shift % axis_len as isize) + axis_len as isize) as usize % axis_len;
                let mut values = self.values.clone();
                for o in 0..outer {
                    for k in 0..axis_len {
                        let dst_k = (k + s) % axis_len;
                        for i in 0..inner {
                            values[o * axis_len * inner + dst_k * inner + i] =
                                self.values[o * axis_len * inner + k * inner + i];
                        }
                    }
                }
                Ok(Self {
                    shape: self.shape.clone(),
                    values,
                    dtype: self.dtype,
                })
            }
        }
    }

    /// Reverse the order of elements along an axis.
    pub fn flip(&self, axis: Option<isize>) -> Result<Self, UFuncError> {
        match axis {
            None => {
                let mut values = self.values.clone();
                values.reverse();
                Ok(Self {
                    shape: self.shape.clone(),
                    values,
                    dtype: self.dtype,
                })
            }
            Some(ax) => {
                let ax = normalize_axis(ax, self.shape.len())?;
                let inner: usize = self.shape[ax + 1..].iter().copied().product();
                let outer: usize = self.shape[..ax].iter().copied().product();
                let axis_len = self.shape[ax];
                let mut values = self.values.clone();
                for o in 0..outer {
                    for k in 0..axis_len / 2 {
                        let rev_k = axis_len - 1 - k;
                        for i in 0..inner {
                            let a = o * axis_len * inner + k * inner + i;
                            let b = o * axis_len * inner + rev_k * inner + i;
                            values.swap(a, b);
                        }
                    }
                }
                Ok(Self {
                    shape: self.shape.clone(),
                    values,
                    dtype: self.dtype,
                })
            }
        }
    }

    // ── Dot product / matrix multiplication ────────────────────────

    /// Compute the dot product of two arrays.
    /// - Both 1-D: inner product (scalar).
    /// - Both 2-D: matrix multiplication.
    /// - 1-D and 2-D: treat 1-D as row/column vector as appropriate.
    pub fn dot(&self, rhs: &Self) -> Result<Self, UFuncError> {
        let ld = self.shape.len();
        let rd = rhs.shape.len();
        match (ld, rd) {
            (1, 1) => {
                // Inner product
                if self.shape[0] != rhs.shape[0] {
                    return Err(UFuncError::Msg(format!(
                        "dot: shapes ({},) and ({},) not aligned",
                        self.shape[0], rhs.shape[0]
                    )));
                }
                let sum: f64 = self
                    .values
                    .iter()
                    .zip(&rhs.values)
                    .map(|(a, b)| a * b)
                    .sum();
                let dtype = promote(self.dtype, rhs.dtype);
                Ok(Self::scalar(sum, dtype))
            }
            (2, 2) => self.matmul(rhs),
            (1, 2) => {
                // Treat 1-D as (1, K), result is (N,)
                let k = self.shape[0];
                if k != rhs.shape[0] {
                    return Err(UFuncError::Msg(format!(
                        "dot: shapes ({k},) and {:?} not aligned: {k} != {}",
                        rhs.shape, rhs.shape[0]
                    )));
                }
                let n = rhs.shape[1];
                let mut values = vec![0.0f64; n];
                for (j, val) in values.iter_mut().enumerate() {
                    let mut acc = 0.0;
                    for i in 0..k {
                        acc += self.values[i] * rhs.values[i * n + j];
                    }
                    *val = acc;
                }
                let dtype = promote(self.dtype, rhs.dtype);
                Ok(Self {
                    shape: vec![n],
                    values,
                    dtype,
                })
            }
            (2, 1) => {
                // Treat 1-D as (K, 1), result is (M,)
                let (m, k) = (self.shape[0], self.shape[1]);
                if k != rhs.shape[0] {
                    return Err(UFuncError::Msg(format!(
                        "dot: shapes {:?} and ({},) not aligned: {k} != {}",
                        self.shape, rhs.shape[0], rhs.shape[0]
                    )));
                }
                let mut values = vec![0.0f64; m];
                for (i, val) in values.iter_mut().enumerate() {
                    let mut acc = 0.0;
                    for j in 0..k {
                        acc += self.values[i * k + j] * rhs.values[j];
                    }
                    *val = acc;
                }
                let dtype = promote(self.dtype, rhs.dtype);
                Ok(Self {
                    shape: vec![m],
                    values,
                    dtype,
                })
            }
            _ => Err(UFuncError::Msg(format!(
                "dot: unsupported shapes {:?} and {:?}",
                self.shape, rhs.shape
            ))),
        }
    }

    /// Matrix multiplication (2-D x 2-D).
    pub fn matmul(&self, rhs: &Self) -> Result<Self, UFuncError> {
        if self.shape.len() != 2 || rhs.shape.len() != 2 {
            return Err(UFuncError::Msg("matmul requires 2-D arrays".to_string()));
        }
        let (m, k1) = (self.shape[0], self.shape[1]);
        let (k2, n) = (rhs.shape[0], rhs.shape[1]);
        if k1 != k2 {
            return Err(UFuncError::Msg(format!(
                "matmul: shapes ({m},{k1}) and ({k2},{n}) not aligned: {k1} != {k2}"
            )));
        }
        let k = k1;
        let mut values = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += self.values[i * k + p] * rhs.values[p * n + j];
                }
                values[i * n + j] = acc;
            }
        }
        let dtype = promote(self.dtype, rhs.dtype);
        Ok(Self {
            shape: vec![m, n],
            values,
            dtype,
        })
    }

    /// Compute the outer product of two 1-D arrays.
    pub fn outer(&self, rhs: &Self) -> Result<Self, UFuncError> {
        if self.shape.len() != 1 || rhs.shape.len() != 1 {
            return Err(UFuncError::Msg("outer requires 1-D arrays".to_string()));
        }
        let m = self.shape[0];
        let n = rhs.shape[0];
        let mut values = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                values[i * n + j] = self.values[i] * rhs.values[j];
            }
        }
        let dtype = promote(self.dtype, rhs.dtype);
        Ok(Self {
            shape: vec![m, n],
            values,
            dtype,
        })
    }

    /// Compute the inner product of two arrays. For 1-D, same as dot.
    pub fn inner(&self, rhs: &Self) -> Result<Self, UFuncError> {
        if self.shape.len() == 1 && rhs.shape.len() == 1 {
            self.dot(rhs)
        } else {
            Err(UFuncError::Msg(
                "inner: only 1-D arrays supported currently".to_string(),
            ))
        }
    }

    /// Compute the trace of a 2-D matrix.
    pub fn trace(&self, offset: i64) -> Result<Self, UFuncError> {
        let d = self.diag(offset)?;
        let sum: f64 = d.values.iter().sum();
        Ok(Self::scalar(sum, self.dtype))
    }

    // ── tensor operations ────────

    /// Kronecker product of two arrays (np.kron).
    pub fn kron(&self, other: &Self) -> Result<Self, UFuncError> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(UFuncError::Msg(
                "kron: only 2-D arrays supported".to_string(),
            ));
        }
        let (m, n) = (self.shape[0], self.shape[1]);
        let (p, q) = (other.shape[0], other.shape[1]);
        let out_rows = m * p;
        let out_cols = n * q;
        let mut values = vec![0.0; out_rows * out_cols];
        for i in 0..m {
            for j in 0..n {
                let a_val = self.values[i * n + j];
                for k in 0..p {
                    for l in 0..q {
                        values[(i * p + k) * out_cols + (j * q + l)] =
                            a_val * other.values[k * q + l];
                    }
                }
            }
        }
        let dtype = promote(self.dtype, other.dtype);
        Ok(Self {
            shape: vec![out_rows, out_cols],
            values,
            dtype,
        })
    }

    /// Tensor dot product (np.tensordot with axes=integer).
    /// Contracts the last `axes` dimensions of `self` with the first `axes` dimensions of `other`.
    pub fn tensordot(&self, other: &Self, axes: usize) -> Result<Self, UFuncError> {
        let a_ndim = self.shape.len();
        let b_ndim = other.shape.len();
        if axes > a_ndim || axes > b_ndim {
            return Err(UFuncError::Msg(
                "tensordot: axes exceeds dimensions".to_string(),
            ));
        }
        // Check contracted dimensions match
        for i in 0..axes {
            if self.shape[a_ndim - axes + i] != other.shape[i] {
                return Err(UFuncError::Msg(
                    "tensordot: shape mismatch on contracted axis".to_string(),
                ));
            }
        }
        let contract_size: usize = self.shape[a_ndim - axes..].iter().product();
        let a_outer: usize = self.shape[..a_ndim - axes].iter().product();
        let b_outer: usize = other.shape[axes..].iter().product();
        let mut values = Vec::with_capacity(a_outer * b_outer);
        for i in 0..a_outer {
            for j in 0..b_outer {
                let mut sum = 0.0;
                for k in 0..contract_size {
                    sum += self.values[i * contract_size + k] * other.values[k * b_outer + j];
                }
                values.push(sum);
            }
        }
        let mut out_shape: Vec<usize> = self.shape[..a_ndim - axes].to_vec();
        out_shape.extend_from_slice(&other.shape[axes..]);
        if out_shape.is_empty() {
            out_shape.push(1);
        }
        let dtype = promote(self.dtype, other.dtype);
        Ok(Self {
            shape: out_shape,
            values,
            dtype,
        })
    }

    /// Vector dot product (np.vdot). Flattens both inputs and computes dot product.
    pub fn vdot(&self, other: &Self) -> Result<Self, UFuncError> {
        if self.values.len() != other.values.len() {
            return Err(UFuncError::Msg(format!(
                "vdot: input sizes must match, got {} and {}",
                self.values.len(),
                other.values.len()
            )));
        }
        let sum: f64 = self
            .values
            .iter()
            .zip(&other.values)
            .map(|(a, b)| a * b)
            .sum();
        let dtype = promote(self.dtype, other.dtype);
        Ok(Self::scalar(sum, dtype))
    }

    /// Compute the dot product of two or more arrays in an optimized order (np.multi_dot).
    /// Uses simple left-to-right evaluation (optimal ordering not yet implemented).
    pub fn multi_dot(arrays: &[&Self]) -> Result<Self, UFuncError> {
        if arrays.len() < 2 {
            return Err(UFuncError::Msg(
                "multi_dot: need at least 2 arrays".to_string(),
            ));
        }
        let mut result = arrays[0].dot(arrays[1])?;
        for arr in &arrays[2..] {
            result = result.dot(arr)?;
        }
        Ok(result)
    }

    // ── Fancy / boolean indexing ──────────────────────────────────

    /// Select elements by integer indices along an axis (np.take).
    /// Negative indices are supported (wrap around).
    pub fn take(&self, indices: &[i64], axis: Option<isize>) -> Result<Self, UFuncError> {
        match axis {
            None => {
                // Flat indexing
                let n = self.values.len() as i64;
                let mut out = Vec::with_capacity(indices.len());
                for &idx in indices {
                    let i = if idx < 0 { idx + n } else { idx };
                    if i < 0 || i >= n {
                        return Err(UFuncError::Msg(format!(
                            "take: index {idx} out of bounds for size {n}"
                        )));
                    }
                    out.push(self.values[i as usize]);
                }
                Ok(Self {
                    shape: vec![out.len()],
                    values: out,
                    dtype: self.dtype,
                })
            }
            Some(ax) => {
                let ax = normalize_axis(ax, self.shape.len())?;
                let axis_len = self.shape[ax] as i64;
                // Resolve and validate indices
                let resolved: Vec<usize> = indices
                    .iter()
                    .map(|&idx| {
                        let i = if idx < 0 { idx + axis_len } else { idx };
                        if i < 0 || i >= axis_len {
                            Err(UFuncError::Msg(format!(
                                "take: index {idx} out of bounds for axis {ax} with size {axis_len}"
                            )))
                        } else {
                            Ok(i as usize)
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let mut out_shape = self.shape.clone();
                out_shape[ax] = resolved.len();

                let outer: usize = self.shape[..ax].iter().product();
                let inner: usize = self.shape[ax + 1..].iter().product();
                let src_stride = self.shape[ax] * inner;

                let mut values = Vec::with_capacity(outer * resolved.len() * inner);
                for o in 0..outer {
                    for &ri in &resolved {
                        let base = o * src_stride + ri * inner;
                        values.extend_from_slice(&self.values[base..base + inner]);
                    }
                }
                Ok(Self {
                    shape: out_shape,
                    values,
                    dtype: self.dtype,
                })
            }
        }
    }

    /// Boolean mask indexing — select elements where mask is true (non-zero).
    /// When axis is None, flattens both arrays. When axis is given, selects
    /// whole slices along that axis where the corresponding mask element is true.
    pub fn compress(&self, condition: &[bool], axis: Option<isize>) -> Result<Self, UFuncError> {
        match axis {
            None => {
                // Flat: condition length must match total elements
                if condition.len() != self.values.len() {
                    return Err(UFuncError::Msg(format!(
                        "compress: condition length {} != array size {}",
                        condition.len(),
                        self.values.len()
                    )));
                }
                let values: Vec<f64> = self
                    .values
                    .iter()
                    .zip(condition)
                    .filter(|&(_, &c)| c)
                    .map(|(&v, _)| v)
                    .collect();
                Ok(Self {
                    shape: vec![values.len()],
                    values,
                    dtype: self.dtype,
                })
            }
            Some(ax) => {
                let ax = normalize_axis(ax, self.shape.len())?;
                let axis_len = self.shape[ax];
                if condition.len() != axis_len {
                    return Err(UFuncError::Msg(format!(
                        "compress: condition length {} != axis size {}",
                        condition.len(),
                        axis_len
                    )));
                }
                let selected: Vec<usize> = condition
                    .iter()
                    .enumerate()
                    .filter(|&(_, &c)| c)
                    .map(|(i, _)| i)
                    .collect();
                let indices: Vec<i64> = selected.iter().map(|&i| i as i64).collect();
                self.take(&indices, Some(ax as isize))
            }
        }
    }

    /// Advanced integer array indexing (flat). Equivalent to `a.flat[indices]`.
    pub fn fancy_index(&self, indices: &[i64]) -> Result<Self, UFuncError> {
        self.take(indices, None)
    }

    /// Boolean mask indexing (flat). Equivalent to `a[mask]`.
    pub fn boolean_index(&self, mask: &Self) -> Result<Self, UFuncError> {
        // mask values: 0.0 = false, anything else = true
        if mask.values.len() != self.values.len() {
            return Err(UFuncError::Msg(format!(
                "boolean_index: mask size {} != array size {}",
                mask.values.len(),
                self.values.len()
            )));
        }
        let condition: Vec<bool> = mask.values.iter().map(|&v| v != 0.0).collect();
        self.compress(&condition, None)
    }

    /// Set elements by boolean mask (flat). Equivalent to `a[mask] = value`.
    pub fn boolean_set(&mut self, mask: &Self, value: f64) -> Result<(), UFuncError> {
        if mask.values.len() != self.values.len() {
            return Err(UFuncError::Msg(format!(
                "boolean_set: mask size {} != array size {}",
                mask.values.len(),
                self.values.len()
            )));
        }
        for (v, &m) in self.values.iter_mut().zip(&mask.values) {
            if m != 0.0 {
                *v = value;
            }
        }
        Ok(())
    }

    /// Set elements by integer indices (flat). Equivalent to `a.flat[indices] = values`.
    pub fn fancy_set(&mut self, indices: &[i64], values: &[f64]) -> Result<(), UFuncError> {
        if indices.len() != values.len() {
            return Err(UFuncError::Msg(format!(
                "fancy_set: indices length {} != values length {}",
                indices.len(),
                values.len()
            )));
        }
        let n = self.values.len() as i64;
        for (&idx, &val) in indices.iter().zip(values) {
            let i = if idx < 0 { idx + n } else { idx };
            if i < 0 || i >= n {
                return Err(UFuncError::Msg(format!(
                    "fancy_set: index {idx} out of bounds for size {n}"
                )));
            }
            self.values[i as usize] = val;
        }
        Ok(())
    }

    // ── advanced indexing ────────

    /// Take values along an axis using index array (np.take_along_axis).
    /// `indices` must have the same number of dimensions as `self`.
    pub fn take_along_axis(&self, indices: &Self, axis: isize) -> Result<Self, UFuncError> {
        let ndim = self.shape.len();
        if indices.shape.len() != ndim {
            return Err(UFuncError::Msg(format!(
                "take_along_axis: indices ndim {} != array ndim {ndim}",
                indices.shape.len()
            )));
        }
        let ax = normalize_axis(axis, ndim)?;
        let axis_len = self.shape[ax];
        let strides = c_strides_elems(&self.shape);
        let idx_strides = c_strides_elems(&indices.shape);
        let total: usize = indices.shape.iter().product();
        let mut values = Vec::with_capacity(total);
        for flat in 0..total {
            // Compute multi-dimensional index into indices array
            let mut rem = flat;
            let mut src_flat = 0usize;
            for d in 0..ndim {
                let coord = rem / idx_strides[d];
                rem %= idx_strides[d];
                if d == ax {
                    let idx = indices.values[flat] as i64;
                    let resolved = if idx < 0 { idx + axis_len as i64 } else { idx };
                    if resolved < 0 || resolved >= axis_len as i64 {
                        return Err(UFuncError::Msg(format!(
                            "take_along_axis: index {idx} out of bounds for axis {ax} with size {axis_len}"
                        )));
                    }
                    src_flat += resolved as usize * strides[d];
                } else {
                    src_flat += coord * strides[d];
                }
            }
            values.push(self.values[src_flat]);
        }
        Ok(Self {
            shape: indices.shape.clone(),
            values,
            dtype: self.dtype,
        })
    }

    /// Put values along an axis using index array (np.put_along_axis).
    pub fn put_along_axis(
        &mut self,
        indices: &Self,
        values: &Self,
        axis: isize,
    ) -> Result<(), UFuncError> {
        let ndim = self.shape.len();
        if indices.shape.len() != ndim || values.shape.len() != ndim {
            return Err(UFuncError::Msg(
                "put_along_axis: all arrays must have same ndim".to_string(),
            ));
        }
        let ax = normalize_axis(axis, ndim)?;
        let axis_len = self.shape[ax];
        let strides = c_strides_elems(&self.shape);
        let idx_strides = c_strides_elems(&indices.shape);
        let total: usize = indices.shape.iter().product();
        for flat in 0..total {
            let mut rem = flat;
            let mut dst_flat = 0usize;
            for d in 0..ndim {
                let coord = rem / idx_strides[d];
                rem %= idx_strides[d];
                if d == ax {
                    let idx = indices.values[flat] as i64;
                    let resolved = if idx < 0 { idx + axis_len as i64 } else { idx };
                    if resolved < 0 || resolved >= axis_len as i64 {
                        return Err(UFuncError::Msg(format!(
                            "put_along_axis: index {idx} out of bounds for axis {ax} with size {axis_len}"
                        )));
                    }
                    dst_flat += resolved as usize * strides[d];
                } else {
                    dst_flat += coord * strides[d];
                }
            }
            self.values[dst_flat] = values.values[flat];
        }
        Ok(())
    }

    /// Return elements chosen from a flattened array based on condition (np.extract).
    pub fn extract(condition: &Self, arr: &Self) -> Result<Self, UFuncError> {
        if condition.values.len() != arr.values.len() {
            return Err(UFuncError::Msg(format!(
                "extract: condition size {} != array size {}",
                condition.values.len(),
                arr.values.len()
            )));
        }
        let values: Vec<f64> = condition
            .values
            .iter()
            .zip(&arr.values)
            .filter(|(c, _)| **c != 0.0)
            .map(|(_, v)| *v)
            .collect();
        let n = values.len();
        Ok(Self {
            shape: vec![n],
            values,
            dtype: arr.dtype,
        })
    }

    /// Change elements of an array based on conditional and input values (np.place).
    /// Values are repeated cyclically if shorter than the number of True entries.
    pub fn place(&mut self, mask: &Self, vals: &[f64]) -> Result<(), UFuncError> {
        if mask.values.len() != self.values.len() {
            return Err(UFuncError::Msg(format!(
                "place: mask size {} != array size {}",
                mask.values.len(),
                self.values.len()
            )));
        }
        if vals.is_empty() {
            return Err(UFuncError::Msg("place: vals must not be empty".to_string()));
        }
        let mut vi = 0;
        for (v, &m) in self.values.iter_mut().zip(&mask.values) {
            if m != 0.0 {
                *v = vals[vi % vals.len()];
                vi += 1;
            }
        }
        Ok(())
    }

    /// Replaces specified flat elements of an array with given values (np.put).
    /// Indices are taken into the flattened array.
    pub fn put(&mut self, indices: &[i64], vals: &[f64]) -> Result<(), UFuncError> {
        if vals.is_empty() && !indices.is_empty() {
            return Err(UFuncError::Msg("put: vals must not be empty".to_string()));
        }
        let n = self.values.len() as i64;
        for (i, &idx) in indices.iter().enumerate() {
            let resolved = if idx < 0 { idx + n } else { idx };
            if resolved < 0 || resolved >= n {
                return Err(UFuncError::Msg(format!(
                    "put: index {idx} out of bounds for size {n}"
                )));
            }
            self.values[resolved as usize] = vals[i % vals.len()];
        }
        Ok(())
    }

    /// Fill the main diagonal of a 2-D array in-place (np.fill_diagonal).
    pub fn fill_diagonal(&mut self, val: f64) -> Result<(), UFuncError> {
        if self.shape.len() != 2 {
            return Err(UFuncError::Msg(
                "fill_diagonal: array must be 2-D".to_string(),
            ));
        }
        let min_dim = self.shape[0].min(self.shape[1]);
        let cols = self.shape[1];
        for i in 0..min_dim {
            self.values[i * cols + i] = val;
        }
        Ok(())
    }

    /// Return the indices of the diagonal elements of a 2-D array (np.diag_indices).
    pub fn diag_indices(n: usize, ndim: usize) -> (Vec<Self>, DType) {
        let idx_values: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let idx = Self {
            shape: vec![n],
            values: idx_values,
            dtype: DType::I64,
        };
        let arrays: Vec<Self> = (0..ndim).map(|_| idx.clone()).collect();
        (arrays, DType::I64)
    }

    /// Return indices for the lower triangle of an (n, m) array (np.tril_indices).
    pub fn tril_indices(n: usize, m: usize, k: i64) -> (Self, Self) {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        for i in 0..n {
            for j in 0..m {
                if (j as i64) <= (i as i64).saturating_add(k) {
                    rows.push(i as f64);
                    cols.push(j as f64);
                }
            }
        }
        let len = rows.len();
        (
            Self {
                shape: vec![len],
                values: rows,
                dtype: DType::I64,
            },
            Self {
                shape: vec![len],
                values: cols,
                dtype: DType::I64,
            },
        )
    }

    /// Return indices for the upper triangle of an (n, m) array (np.triu_indices).
    pub fn triu_indices(n: usize, m: usize, k: i64) -> (Self, Self) {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        for i in 0..n {
            for j in 0..m {
                if (j as i64) >= (i as i64).saturating_add(k) {
                    rows.push(i as f64);
                    cols.push(j as f64);
                }
            }
        }
        let len = rows.len();
        (
            Self {
                shape: vec![len],
                values: rows,
                dtype: DType::I64,
            },
            Self {
                shape: vec![len],
                values: cols,
                dtype: DType::I64,
            },
        )
    }

    // ── Type casting, boolean reductions, searching, comparison ────

    /// Cast the array to a different dtype (np.ndarray.astype).
    /// Values are reinterpreted: for integer dtypes, values are truncated;
    /// for bool, non-zero becomes 1.0.
    pub fn astype(&self, dtype: DType) -> Self {
        let values: Vec<f64> = match dtype {
            DType::Bool => self
                .values
                .iter()
                .map(|&v| if v != 0.0 { 1.0 } else { 0.0 })
                .collect(),
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                self.values.iter().map(|&v| (v as i64) as f64).collect()
            }
            DType::U8 | DType::U16 | DType::U32 | DType::U64 => {
                self.values.iter().map(|&v| (v as u64) as f64).collect()
            }
            DType::F32 => self.values.iter().map(|&v| (v as f32) as f64).collect(),
            DType::F64 => self.values.clone(),
        };
        Self {
            shape: self.shape.clone(),
            values,
            dtype,
        }
    }

    /// Test whether any element evaluates to true (non-zero).
    /// With axis=None, returns a scalar. With axis, reduces along that axis.
    pub fn any(&self, axis: Option<isize>) -> Result<Self, UFuncError> {
        match axis {
            None => {
                let result = if self.values.iter().any(|&v| v != 0.0) {
                    1.0
                } else {
                    0.0
                };
                Ok(Self::scalar(result, DType::Bool))
            }
            Some(ax) => {
                let ax = normalize_axis(ax, self.shape.len())?;
                let axis_len = self.shape[ax];
                let outer: usize = self.shape[..ax].iter().product();
                let inner: usize = self.shape[ax + 1..].iter().product();
                let mut out_shape = self.shape.clone();
                out_shape.remove(ax);
                if out_shape.is_empty() {
                    out_shape.push(1);
                }
                let mut values = Vec::with_capacity(outer * inner);
                for o in 0..outer {
                    for i in 0..inner {
                        let found = (0..axis_len)
                            .any(|a| self.values[o * axis_len * inner + a * inner + i] != 0.0);
                        values.push(if found { 1.0 } else { 0.0 });
                    }
                }
                Ok(Self {
                    shape: out_shape,
                    values,
                    dtype: DType::Bool,
                })
            }
        }
    }

    /// Test whether all elements evaluate to true (non-zero).
    /// With axis=None, returns a scalar. With axis, reduces along that axis.
    pub fn all(&self, axis: Option<isize>) -> Result<Self, UFuncError> {
        match axis {
            None => {
                let result = if self.values.iter().all(|&v| v != 0.0) {
                    1.0
                } else {
                    0.0
                };
                Ok(Self::scalar(result, DType::Bool))
            }
            Some(ax) => {
                let ax = normalize_axis(ax, self.shape.len())?;
                let axis_len = self.shape[ax];
                let outer: usize = self.shape[..ax].iter().product();
                let inner: usize = self.shape[ax + 1..].iter().product();
                let mut out_shape = self.shape.clone();
                out_shape.remove(ax);
                if out_shape.is_empty() {
                    out_shape.push(1);
                }
                let mut values = Vec::with_capacity(outer * inner);
                for o in 0..outer {
                    for i in 0..inner {
                        let ok = (0..axis_len)
                            .all(|a| self.values[o * axis_len * inner + a * inner + i] != 0.0);
                        values.push(if ok { 1.0 } else { 0.0 });
                    }
                }
                Ok(Self {
                    shape: out_shape,
                    values,
                    dtype: DType::Bool,
                })
            }
        }
    }

    /// Return the indices of non-zero (truthy) elements (flat).
    /// Returns a 1-D array of integer indices. Equivalent to `np.nonzero` on flat.
    pub fn nonzero(&self) -> Self {
        let indices: Vec<f64> = self
            .values
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v != 0.0)
            .map(|(i, _)| i as f64)
            .collect();
        let n = indices.len();
        Self {
            shape: vec![n],
            values: indices,
            dtype: DType::I64,
        }
    }

    /// Calculate the n-th discrete difference along the given axis (np.diff).
    /// Default n=1 and axis=-1.
    pub fn diff(&self, n: usize, axis: Option<isize>) -> Result<Self, UFuncError> {
        if n == 0 {
            return Ok(self.clone());
        }
        let ax = match axis {
            Some(a) => normalize_axis(a, self.shape.len())?,
            None => self.shape.len() - 1,
        };
        if self.shape[ax] < 1 {
            return Err(UFuncError::Msg(
                "diff: axis length must be >= 1".to_string(),
            ));
        }

        let mut current = self.clone();
        for _ in 0..n {
            let axis_len = current.shape[ax];
            if axis_len < 1 {
                break;
            }
            let outer: usize = current.shape[..ax].iter().product();
            let inner: usize = current.shape[ax + 1..].iter().product();
            let new_axis_len = axis_len.saturating_sub(1);
            let mut out_shape = current.shape.clone();
            out_shape[ax] = new_axis_len;
            let mut values = Vec::with_capacity(outer * new_axis_len * inner);
            for o in 0..outer {
                for a in 0..new_axis_len {
                    for i in 0..inner {
                        let base = o * axis_len * inner;
                        let v1 = current.values[base + (a + 1) * inner + i];
                        let v0 = current.values[base + a * inner + i];
                        values.push(v1 - v0);
                    }
                }
            }
            current = Self {
                shape: out_shape,
                values,
                dtype: current.dtype,
            };
        }
        Ok(current)
    }

    /// Element-wise comparison: |a - b| <= atol + rtol * |b| (np.isclose).
    /// Returns a boolean array.
    pub fn isclose(&self, other: &Self, rtol: f64, atol: f64) -> Result<Self, UFuncError> {
        if self.values.len() != other.values.len() {
            return Err(UFuncError::Msg(format!(
                "isclose: size mismatch {} vs {}",
                self.values.len(),
                other.values.len()
            )));
        }
        let values: Vec<f64> = self
            .values
            .iter()
            .zip(&other.values)
            .map(|(&a, &b)| {
                if (a - b).abs() <= atol + rtol * b.abs() {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        Ok(Self {
            shape: self.shape.clone(),
            values,
            dtype: DType::Bool,
        })
    }

    /// Returns true if two arrays are element-wise equal within tolerance (np.allclose).
    pub fn allclose(&self, other: &Self, rtol: f64, atol: f64) -> Result<bool, UFuncError> {
        let close = self.isclose(other, rtol, atol)?;
        Ok(close.values.iter().all(|&v| v != 0.0))
    }

    // ── Median, percentile, cummin, cummax, pad ───────────────────

    /// Compute the median along the given axis (or all elements if None).
    pub fn median(&self, axis: Option<isize>) -> Result<Self, UFuncError> {
        match axis {
            None => {
                let mut sorted = self.values.clone();
                sorted.sort_by(|a, b| a.total_cmp(b));
                let n = sorted.len();
                if n == 0 {
                    return Err(UFuncError::Msg("median of empty array".to_string()));
                }
                let med = if n % 2 == 1 {
                    sorted[n / 2]
                } else {
                    (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
                };
                Ok(Self::scalar(med, DType::F64))
            }
            Some(ax) => {
                let ax = normalize_axis(ax, self.shape.len())?;
                let axis_len = self.shape[ax];
                if axis_len == 0 {
                    return Err(UFuncError::Msg("median of zero-length axis".to_string()));
                }
                let outer: usize = self.shape[..ax].iter().product();
                let inner: usize = self.shape[ax + 1..].iter().product();
                let mut out_shape = self.shape.clone();
                out_shape.remove(ax);
                if out_shape.is_empty() {
                    out_shape.push(1);
                }
                let mut values = Vec::with_capacity(outer * inner);
                for o in 0..outer {
                    for i in 0..inner {
                        let mut lane: Vec<f64> = (0..axis_len)
                            .map(|a| self.values[o * axis_len * inner + a * inner + i])
                            .collect();
                        lane.sort_by(|a, b| a.total_cmp(b));
                        let med = if axis_len % 2 == 1 {
                            lane[axis_len / 2]
                        } else {
                            (lane[axis_len / 2 - 1] + lane[axis_len / 2]) / 2.0
                        };
                        values.push(med);
                    }
                }
                Ok(Self {
                    shape: out_shape,
                    values,
                    dtype: DType::F64,
                })
            }
        }
    }

    /// Compute the q-th percentile of the data along the given axis.
    /// q is in [0, 100]. Uses linear interpolation (np default).
    pub fn percentile(&self, q: f64, axis: Option<isize>) -> Result<Self, UFuncError> {
        if !(0.0..=100.0).contains(&q) {
            return Err(UFuncError::Msg(format!(
                "percentile: q={q} must be in [0, 100]"
            )));
        }
        let fraction = q / 100.0;
        match axis {
            None => {
                let mut sorted = self.values.clone();
                sorted.sort_by(|a, b| a.total_cmp(b));
                let n = sorted.len();
                if n == 0 {
                    return Err(UFuncError::Msg("percentile of empty array".to_string()));
                }
                let val = interpolate_percentile(&sorted, fraction);
                Ok(Self::scalar(val, DType::F64))
            }
            Some(ax) => {
                let ax = normalize_axis(ax, self.shape.len())?;
                let axis_len = self.shape[ax];
                if axis_len == 0 {
                    return Err(UFuncError::Msg(
                        "percentile of zero-length axis".to_string(),
                    ));
                }
                let outer: usize = self.shape[..ax].iter().product();
                let inner: usize = self.shape[ax + 1..].iter().product();
                let mut out_shape = self.shape.clone();
                out_shape.remove(ax);
                if out_shape.is_empty() {
                    out_shape.push(1);
                }
                let mut values = Vec::with_capacity(outer * inner);
                for o in 0..outer {
                    for i in 0..inner {
                        let mut lane: Vec<f64> = (0..axis_len)
                            .map(|a| self.values[o * axis_len * inner + a * inner + i])
                            .collect();
                        lane.sort_by(|a, b| a.total_cmp(b));
                        values.push(interpolate_percentile(&lane, fraction));
                    }
                }
                Ok(Self {
                    shape: out_shape,
                    values,
                    dtype: DType::F64,
                })
            }
        }
    }

    /// Central moment of order `order` over the flattened array
    /// (scipy.stats.moment).
    pub fn moment(&self, order: usize) -> Result<f64, UFuncError> {
        let n = self.values.len();
        if n == 0 {
            return Err(UFuncError::Msg("moment of empty array".to_string()));
        }
        if order == 0 {
            return Ok(1.0);
        }
        let mean = self.values.iter().sum::<f64>() / n as f64;
        let m = self
            .values
            .iter()
            .map(|&v| (v - mean).powi(order as i32))
            .sum::<f64>()
            / n as f64;
        Ok(m)
    }

    /// Fisher-Pearson coefficient of skewness (scipy.stats.skew).
    /// skew = m3 / m2^(3/2) where m_k is the k-th central moment.
    pub fn skew(&self) -> Result<f64, UFuncError> {
        let m2 = self.moment(2)?;
        if m2 <= f64::EPSILON {
            return Ok(0.0);
        }
        let m3 = self.moment(3)?;
        Ok(m3 / m2.powf(1.5))
    }

    /// Excess kurtosis (scipy.stats.kurtosis, Fisher definition).
    /// kurtosis = m4 / m2^2 - 3.
    pub fn kurtosis(&self) -> Result<f64, UFuncError> {
        let m2 = self.moment(2)?;
        if m2 <= f64::EPSILON {
            return Ok(0.0);
        }
        let m4 = self.moment(4)?;
        Ok(m4 / (m2 * m2) - 3.0)
    }

    /// Modal value (most frequent) of the array (scipy.stats.mode equivalent).
    /// Returns (mode_value, count). For ties, returns the smallest value.
    pub fn mode(&self) -> Result<(f64, usize), UFuncError> {
        if self.values.is_empty() {
            return Err(UFuncError::Msg("mode: empty array".to_string()));
        }
        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut best_val = sorted[0];
        let mut best_count = 1_usize;
        let mut cur_val = sorted[0];
        let mut cur_count = 1_usize;
        for &v in &sorted[1..] {
            if (v - cur_val).abs() < f64::EPSILON {
                cur_count += 1;
            } else {
                if cur_count > best_count {
                    best_val = cur_val;
                    best_count = cur_count;
                }
                cur_val = v;
                cur_count = 1;
            }
        }
        if cur_count > best_count {
            best_val = cur_val;
            best_count = cur_count;
        }
        Ok((best_val, best_count))
    }

    /// Shannon entropy of the array values (scipy.stats.entropy equivalent).
    /// Treats values as probabilities (must be non-negative).
    /// Returns -sum(p * ln(p)) for p > 0.
    pub fn entropy(&self) -> Result<f64, UFuncError> {
        if self.values.is_empty() {
            return Err(UFuncError::Msg("entropy: empty array".to_string()));
        }
        let total: f64 = self.values.iter().sum();
        if total <= 0.0 {
            return Err(UFuncError::Msg(
                "entropy: sum of values must be positive".to_string(),
            ));
        }
        let h = self.values.iter().fold(0.0, |acc, &v| {
            if v > 0.0 {
                let p = v / total;
                acc - p * p.ln()
            } else {
                acc
            }
        });
        Ok(h)
    }

    /// Cumulative minimum along the given axis (or flat if None).
    pub fn cummin(&self, axis: Option<isize>) -> Result<Self, UFuncError> {
        self.cumulative_op(axis, |running, v| if v < running { v } else { running })
    }

    /// Cumulative maximum along the given axis (or flat if None).
    pub fn cummax(&self, axis: Option<isize>) -> Result<Self, UFuncError> {
        self.cumulative_op(axis, |running, v| if v > running { v } else { running })
    }

    /// Generic cumulative operation along an axis.
    fn cumulative_op(
        &self,
        axis: Option<isize>,
        op: impl Fn(f64, f64) -> f64,
    ) -> Result<Self, UFuncError> {
        match axis {
            None => {
                if self.values.is_empty() {
                    return Ok(self.clone());
                }
                let mut values = Vec::with_capacity(self.values.len());
                let mut acc = self.values[0];
                values.push(acc);
                for &v in &self.values[1..] {
                    acc = op(acc, v);
                    values.push(acc);
                }
                Ok(Self {
                    shape: self.shape.clone(),
                    values,
                    dtype: self.dtype,
                })
            }
            Some(ax) => {
                let ax = normalize_axis(ax, self.shape.len())?;
                let axis_len = self.shape[ax];
                if axis_len == 0 {
                    return Ok(self.clone());
                }
                let outer: usize = self.shape[..ax].iter().product();
                let inner: usize = self.shape[ax + 1..].iter().product();
                let mut values = vec![0.0f64; self.values.len()];
                for o in 0..outer {
                    for i in 0..inner {
                        let base = o * axis_len * inner;
                        let mut acc = self.values[base + i];
                        values[base + i] = acc;
                        for a in 1..axis_len {
                            let idx = base + a * inner + i;
                            acc = op(acc, self.values[idx]);
                            values[idx] = acc;
                        }
                    }
                }
                Ok(Self {
                    shape: self.shape.clone(),
                    values,
                    dtype: self.dtype,
                })
            }
        }
    }

    /// Pad an array with constant values (np.pad with mode='constant').
    /// `pad_width` is a list of (before, after) tuples, one per dimension.
    pub fn pad(&self, pad_width: &[(usize, usize)], constant: f64) -> Result<Self, UFuncError> {
        let ndim = self.shape.len();
        if pad_width.len() != ndim {
            return Err(UFuncError::Msg(format!(
                "pad: pad_width length {} != ndim {}",
                pad_width.len(),
                ndim
            )));
        }
        let out_shape: Vec<usize> = self
            .shape
            .iter()
            .zip(pad_width)
            .map(|(&s, &(b, a))| s + b + a)
            .collect();
        let out_count: usize = out_shape.iter().product();
        let src_strides = c_strides_elems(&self.shape);
        let out_strides = c_strides_elems(&out_shape);
        let mut values = vec![constant; out_count];

        // Copy source values into the padded region
        let src_count: usize = self.shape.iter().product();
        for flat in 0..src_count {
            let mut remainder = flat;
            let mut out_flat = 0;
            for d in 0..ndim {
                let idx = remainder / src_strides[d];
                remainder %= src_strides[d];
                out_flat += (idx + pad_width[d].0) * out_strides[d];
            }
            values[out_flat] = self.values[flat];
        }
        Ok(Self {
            shape: out_shape,
            values,
            dtype: self.dtype,
        })
    }

    /// Pad an array with edge values (np.pad with mode='edge').
    pub fn pad_edge(&self, pad_width: &[(usize, usize)]) -> Result<Self, UFuncError> {
        let ndim = self.shape.len();
        if pad_width.len() != ndim {
            return Err(UFuncError::Msg(format!(
                "pad_edge: pad_width length {} != ndim {}",
                pad_width.len(),
                ndim
            )));
        }
        let out_shape: Vec<usize> = self
            .shape
            .iter()
            .zip(pad_width)
            .map(|(&s, &(b, a))| s + b + a)
            .collect();
        let out_count: usize = out_shape.iter().product();
        let src_strides = c_strides_elems(&self.shape);
        let out_strides = c_strides_elems(&out_shape);
        let values: Vec<f64> = (0..out_count)
            .map(|out_flat| {
                let mut remainder = out_flat;
                let mut src_flat = 0;
                for d in 0..ndim {
                    let out_idx = remainder / out_strides[d];
                    remainder %= out_strides[d];
                    let src_idx = if out_idx < pad_width[d].0 {
                        0
                    } else if out_idx >= pad_width[d].0 + self.shape[d] {
                        self.shape[d].saturating_sub(1)
                    } else {
                        out_idx - pad_width[d].0
                    };
                    src_flat += src_idx * src_strides[d];
                }
                self.values[src_flat]
            })
            .collect();
        Ok(Self {
            shape: out_shape,
            values,
            dtype: self.dtype,
        })
    }

    /// Pad an array with wrapped values (np.pad with mode='wrap').
    pub fn pad_wrap(&self, pad_width: &[(usize, usize)]) -> Result<Self, UFuncError> {
        let ndim = self.shape.len();
        if pad_width.len() != ndim {
            return Err(UFuncError::Msg(format!(
                "pad_wrap: pad_width length {} != ndim {}",
                pad_width.len(),
                ndim
            )));
        }
        for (d, (&s, &(b, a))) in self.shape.iter().zip(pad_width).enumerate() {
            if s == 0 && (b > 0 || a > 0) {
                return Err(UFuncError::Msg(format!(
                    "pad_wrap: axis {d} has size 0, cannot wrap"
                )));
            }
        }
        let out_shape: Vec<usize> = self
            .shape
            .iter()
            .zip(pad_width)
            .map(|(&s, &(b, a))| s + b + a)
            .collect();
        let out_count: usize = out_shape.iter().product();
        let src_strides = c_strides_elems(&self.shape);
        let out_strides = c_strides_elems(&out_shape);
        let values: Vec<f64> = (0..out_count)
            .map(|out_flat| {
                let mut remainder = out_flat;
                let mut src_flat = 0;
                for d in 0..ndim {
                    let out_idx = remainder / out_strides[d];
                    remainder %= out_strides[d];
                    let shifted = out_idx as isize - pad_width[d].0 as isize;
                    let src_idx = shifted.rem_euclid(self.shape[d] as isize) as usize;
                    src_flat += src_idx * src_strides[d];
                }
                self.values[src_flat]
            })
            .collect();
        Ok(Self {
            shape: out_shape,
            values,
            dtype: self.dtype,
        })
    }

    /// Pad an array with reflected values (np.pad with mode='reflect').
    /// Reflects without duplicating the edge value.
    pub fn pad_reflect(&self, pad_width: &[(usize, usize)]) -> Result<Self, UFuncError> {
        let ndim = self.shape.len();
        if pad_width.len() != ndim {
            return Err(UFuncError::Msg(format!(
                "pad_reflect: pad_width length {} != ndim {}",
                pad_width.len(),
                ndim
            )));
        }
        for (d, (&s, &(b, a))) in self.shape.iter().zip(pad_width).enumerate() {
            if s <= 1 && (b > 0 || a > 0) {
                return Err(UFuncError::Msg(format!(
                    "pad_reflect: axis {d} has size {s}, cannot reflect"
                )));
            }
        }
        let out_shape: Vec<usize> = self
            .shape
            .iter()
            .zip(pad_width)
            .map(|(&s, &(b, a))| s + b + a)
            .collect();
        let out_count: usize = out_shape.iter().product();
        let src_strides = c_strides_elems(&self.shape);
        let out_strides = c_strides_elems(&out_shape);
        let values: Vec<f64> = (0..out_count)
            .map(|out_flat| {
                let mut remainder = out_flat;
                let mut src_flat = 0;
                for d in 0..ndim {
                    let out_idx = remainder / out_strides[d];
                    remainder %= out_strides[d];
                    let shifted = out_idx as isize - pad_width[d].0 as isize;
                    let src_idx = reflect_index(shifted, self.shape[d]);
                    src_flat += src_idx * src_strides[d];
                }
                self.values[src_flat]
            })
            .collect();
        Ok(Self {
            shape: out_shape,
            values,
            dtype: self.dtype,
        })
    }

    /// Pad an array with symmetric reflection (np.pad with mode='symmetric').
    /// Reflects including the edge value.
    pub fn pad_symmetric(&self, pad_width: &[(usize, usize)]) -> Result<Self, UFuncError> {
        let ndim = self.shape.len();
        if pad_width.len() != ndim {
            return Err(UFuncError::Msg(format!(
                "pad_symmetric: pad_width length {} != ndim {}",
                pad_width.len(),
                ndim
            )));
        }
        let out_shape: Vec<usize> = self
            .shape
            .iter()
            .zip(pad_width)
            .map(|(&s, &(b, a))| s + b + a)
            .collect();
        let out_count: usize = out_shape.iter().product();
        let src_strides = c_strides_elems(&self.shape);
        let out_strides = c_strides_elems(&out_shape);
        let values: Vec<f64> = (0..out_count)
            .map(|out_flat| {
                let mut remainder = out_flat;
                let mut src_flat = 0;
                for d in 0..ndim {
                    let out_idx = remainder / out_strides[d];
                    remainder %= out_strides[d];
                    let shifted = out_idx as isize - pad_width[d].0 as isize;
                    let src_idx = symmetric_index(shifted, self.shape[d]);
                    src_flat += src_idx * src_strides[d];
                }
                self.values[src_flat]
            })
            .collect();
        Ok(Self {
            shape: out_shape,
            values,
            dtype: self.dtype,
        })
    }

    // ── meshgrid, gradient, histogram, bincount, interp ────────────

    /// Generate coordinate matrices from 1-D coordinate vectors (np.meshgrid).
    /// Returns one array per input, each with shape (len(y), len(x)) for 2 inputs.
    /// Only supports 'xy' indexing (default NumPy).
    pub fn meshgrid(arrays: &[Self]) -> Result<Vec<Self>, UFuncError> {
        if arrays.len() < 2 {
            return Err(UFuncError::Msg(
                "meshgrid requires at least 2 arrays".to_string(),
            ));
        }
        for arr in arrays {
            if arr.shape.len() != 1 {
                return Err(UFuncError::Msg(
                    "meshgrid: all inputs must be 1-D".to_string(),
                ));
            }
        }
        // For 2-D case: shape = (len(arrays[1]), len(arrays[0])) with xy indexing
        // Generalized N-D: output shape = (len(a1), len(a0), len(a2), ...)
        // NumPy xy indexing swaps the first two
        let ndim = arrays.len();
        let mut out_shape: Vec<usize> = arrays.iter().map(|a| a.shape[0]).collect();
        if ndim >= 2 {
            out_shape.swap(0, 1);
        }
        let out_count: usize = out_shape.iter().product();
        let out_strides = c_strides_elems(&out_shape);

        let mut results = Vec::with_capacity(ndim);
        for (dim, arr) in arrays.iter().enumerate() {
            // Which output axis does this input correspond to?
            let axis = if ndim >= 2 {
                match dim {
                    0 => 1,
                    1 => 0,
                    d => d,
                }
            } else {
                dim
            };
            let mut values = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let idx = (flat / out_strides[axis]) % out_shape[axis];
                values.push(arr.values[idx]);
            }
            let dtype = arr.dtype;
            results.push(Self {
                shape: out_shape.clone(),
                values,
                dtype,
            });
        }
        Ok(results)
    }

    /// Compute the numerical gradient along an axis (np.gradient).
    /// Uses central differences for interior points, forward/backward for edges.
    /// With axis=None on 1-D, returns gradient of the flat array.
    /// With axis specified, computes gradient along that axis.
    pub fn gradient(&self) -> Result<Self, UFuncError> {
        self.gradient_axis(None)
    }

    /// Compute gradient along a specific axis (np.gradient with axis parameter).
    pub fn gradient_axis(&self, axis: Option<isize>) -> Result<Self, UFuncError> {
        let ax = match axis {
            None => {
                if self.shape.len() == 1 {
                    0
                } else {
                    return self.gradient_all_axes();
                }
            }
            Some(a) => normalize_axis(a, self.shape.len())?,
        };
        let n = self.shape[ax];
        if n < 2 {
            return Err(UFuncError::Msg(
                "gradient: need at least 2 elements along axis".to_string(),
            ));
        }
        let strides = c_strides_elems(&self.shape);
        let total: usize = self.shape.iter().product();
        let out_strides = c_strides_elems(&self.shape);
        let values: Vec<f64> = (0..total)
            .map(|flat| {
                let mut rem = flat;
                let mut k_val = 0usize;
                for (d, &stride) in out_strides.iter().enumerate() {
                    let coord = rem / stride;
                    rem %= stride;
                    if d == ax {
                        k_val = coord;
                    }
                }
                let base = flat - k_val * strides[ax];
                if k_val == 0 {
                    self.values[base + strides[ax]] - self.values[base]
                } else if k_val == n - 1 {
                    self.values[base + k_val * strides[ax]]
                        - self.values[base + (k_val - 1) * strides[ax]]
                } else {
                    (self.values[base + (k_val + 1) * strides[ax]]
                        - self.values[base + (k_val - 1) * strides[ax]])
                        / 2.0
                }
            })
            .collect();
        Ok(Self {
            shape: self.shape.clone(),
            values,
            dtype: DType::F64,
        })
    }

    /// Compute gradient along all axes, returning the first axis only (for backwards compat).
    fn gradient_all_axes(&self) -> Result<Self, UFuncError> {
        self.gradient_axis(Some(0))
    }

    /// Evaluate a piecewise-defined function (np.piecewise).
    /// `condlist` is a slice of boolean arrays (same shape as self).
    /// `funclist` is a slice of constant values to assign where each condition is true.
    /// If funclist has one extra element, it's used as the default.
    pub fn piecewise(&self, condlist: &[Self], funclist: &[f64]) -> Result<Self, UFuncError> {
        if funclist.len() != condlist.len() && funclist.len() != condlist.len() + 1 {
            return Err(UFuncError::Msg(
                "piecewise: funclist must have same length as condlist or one more".to_string(),
            ));
        }
        let default_val = if funclist.len() == condlist.len() + 1 {
            funclist[condlist.len()]
        } else {
            0.0
        };
        let mut values = vec![default_val; self.values.len()];
        // Apply conditions in reverse order so first condition takes priority
        for (ci, cond) in condlist.iter().enumerate().rev() {
            if cond.values.len() != self.values.len() {
                return Err(UFuncError::Msg(
                    "piecewise: condition shape mismatch".to_string(),
                ));
            }
            for (i, &c) in cond.values.iter().enumerate() {
                if c != 0.0 {
                    values[i] = funclist[ci];
                }
            }
        }
        Ok(Self {
            shape: self.shape.clone(),
            values,
            dtype: self.dtype,
        })
    }

    /// Apply a function along a given axis (np.apply_along_axis).
    /// The function takes a 1-D slice and returns a 1-D array (possibly shorter).
    pub fn apply_along_axis<F>(&self, func: F, axis: isize) -> Result<Self, UFuncError>
    where
        F: Fn(&Self) -> Result<Self, UFuncError>,
    {
        let ax = normalize_axis(axis, self.shape.len())?;
        let axis_len = self.shape[ax];
        let strides = c_strides_elems(&self.shape);
        let outer_count: usize = self
            .shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != ax)
            .map(|(_, &d)| d)
            .product();

        // Extract the first slice to determine output length. Use dummy if empty.
        let first_slice_vals: Vec<f64> = if self.values.is_empty() {
            vec![0.0f64; axis_len]
        } else {
            (0..axis_len)
                .map(|k| self.values[k * strides[ax]])
                .collect()
        };

        let first_slice = Self {
            shape: vec![axis_len],
            values: first_slice_vals,
            dtype: self.dtype,
        };
        let first_result = func(&first_slice)?;
        let result_len = first_result.values.len();

        let mut out_shape = self.shape.clone();
        out_shape[ax] = result_len;

        if self.values.is_empty() {
            return Ok(Self {
                shape: out_shape,
                values: Vec::new(),
                dtype: self.dtype,
            });
        }

        let mut outer_shape = self.shape.clone();
        outer_shape.remove(ax);
        let outer_strides = c_strides_elems(&outer_shape);

        let mut all_values = Vec::with_capacity(outer_count * result_len);
        for outer in 0..outer_count {
            let mut remainder = outer;
            let mut base_flat = 0usize;
            for (outer_dim, &outer_stride) in outer_strides.iter().enumerate() {
                if let Some(coord) = remainder.checked_div(outer_stride) {
                    remainder %= outer_stride;
                    let d = if outer_dim >= ax {
                        outer_dim + 1
                    } else {
                        outer_dim
                    };
                    base_flat += coord * strides[d];
                }
            }
            let slice_vals: Vec<f64> = (0..axis_len)
                .map(|k| self.values[base_flat + k * strides[ax]])
                .collect();
            let slice_arr = Self {
                shape: vec![axis_len],
                values: slice_vals,
                dtype: self.dtype,
            };
            let result = func(&slice_arr)?;
            all_values.extend_from_slice(&result.values);
        }
        Ok(Self {
            shape: out_shape,
            values: all_values,
            dtype: self.dtype,
        })
    }

    /// Compute histogram of a 1-D dataset (np.histogram).
    /// Returns (counts, bin_edges) where counts has length `bins`
    /// and bin_edges has length `bins + 1`.
    pub fn histogram(&self, bins: usize) -> Result<(Self, Self), UFuncError> {
        if self.shape.len() != 1 {
            return Err(UFuncError::Msg(
                "histogram: only 1-D arrays supported".to_string(),
            ));
        }
        if bins == 0 {
            return Err(UFuncError::Msg("histogram: bins must be > 0".to_string()));
        }
        if self.values.is_empty() {
            let counts = Self {
                shape: vec![bins],
                values: vec![0.0; bins],
                dtype: DType::I64,
            };
            let edges = Self::linspace(0.0, 1.0, bins + 1, DType::F64)?;
            return Ok((counts, edges));
        }

        let min = self.values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = self
            .values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let range = if (max - min).abs() < f64::EPSILON {
            1.0
        } else {
            max - min
        };
        let width = range / bins as f64;

        let mut counts = vec![0.0f64; bins];
        for &v in &self.values {
            let idx = ((v - min) / width) as usize;
            let idx = idx.min(bins - 1); // clamp last edge
            counts[idx] += 1.0;
        }

        let mut edges = Vec::with_capacity(bins + 1);
        for i in 0..=bins {
            edges.push(min + i as f64 * width);
        }

        let counts_arr = Self {
            shape: vec![bins],
            values: counts,
            dtype: DType::I64,
        };
        let edges_arr = Self {
            shape: vec![bins + 1],
            values: edges,
            dtype: DType::F64,
        };
        Ok((counts_arr, edges_arr))
    }

    /// Histogram with custom bin edges (np.histogram with array bins).
    /// `bin_edges` must be sorted and 1-D. Returns (counts, bin_edges).
    pub fn histogram_edges(&self, bin_edges: &Self) -> Result<(Self, Self), UFuncError> {
        if self.shape.len() != 1 {
            return Err(UFuncError::Msg(
                "histogram_edges: only 1-D arrays supported".to_string(),
            ));
        }
        if bin_edges.shape.len() != 1 || bin_edges.values.len() < 2 {
            return Err(UFuncError::Msg(
                "histogram_edges: bin_edges must be 1-D with at least 2 elements".to_string(),
            ));
        }
        let n_bins = bin_edges.values.len() - 1;
        let mut counts = vec![0.0f64; n_bins];
        for &v in &self.values {
            // Binary search for bin
            let mut lo = 0;
            let mut hi = n_bins;
            while lo < hi {
                let mid = (lo + hi) / 2;
                if v >= bin_edges.values[mid + 1] {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            if lo < n_bins && v >= bin_edges.values[lo] && v <= bin_edges.values[lo + 1] {
                counts[lo] += 1.0;
            }
        }
        Ok((
            Self {
                shape: vec![n_bins],
                values: counts,
                dtype: DType::I64,
            },
            bin_edges.clone(),
        ))
    }

    /// Histogram with automatic binning strategy (np.histogram with bins='auto'/'sturges'/'sqrt').
    /// Supported strategies: "sturges", "sqrt", "auto" (uses max of sturges and sqrt).
    pub fn histogram_auto(&self, strategy: &str) -> Result<(Self, Self), UFuncError> {
        if self.shape.len() != 1 {
            return Err(UFuncError::Msg(
                "histogram_auto: only 1-D arrays supported".to_string(),
            ));
        }
        let n = self.values.len();
        if n == 0 {
            return self.histogram(10);
        }
        let bins = match strategy {
            "sturges" => ((n as f64).log2().ceil() as usize + 1).max(1),
            "sqrt" => ((n as f64).sqrt().ceil() as usize).max(1),
            "auto" => {
                let sturges = (n as f64).log2().ceil() as usize + 1;
                let sqrt = (n as f64).sqrt().ceil() as usize;
                sturges.max(sqrt).max(1)
            }
            _ => {
                return Err(UFuncError::Msg(format!(
                    "histogram_auto: unknown strategy '{strategy}'"
                )));
            }
        };
        self.histogram(bins)
    }

    /// Count occurrences of each non-negative integer value (np.bincount).
    /// Input values are treated as non-negative integers.
    pub fn bincount(&self) -> Result<Self, UFuncError> {
        if self.shape.len() != 1 {
            return Err(UFuncError::Msg(
                "bincount: only 1-D arrays supported".to_string(),
            ));
        }
        if self.values.is_empty() {
            return Ok(Self {
                shape: vec![0],
                values: Vec::new(),
                dtype: DType::I64,
            });
        }
        let mut max_val = 0usize;
        for &v in &self.values {
            if v < 0.0 {
                return Err(UFuncError::Msg(
                    "bincount: input must be non-negative".to_string(),
                ));
            }
            if v > (isize::MAX as usize / 8) as f64 {
                return Err(UFuncError::Msg(
                    "bincount: input value is too large for memory allocation".to_string(),
                ));
            }
            let iv = v as usize;
            if iv > max_val {
                max_val = iv;
            }
        }
        let len = max_val
            .checked_add(1)
            .ok_or_else(|| UFuncError::Msg("bincount: max value too large".to_string()))?;
        if len > isize::MAX as usize / 8 {
            return Err(UFuncError::Msg(
                "bincount: resulting array size would exceed maximum allowed allocation"
                    .to_string(),
            ));
        }
        let mut counts = vec![0.0f64; len];
        for &v in &self.values {
            counts[v as usize] += 1.0;
        }
        let n = counts.len();
        Ok(Self {
            shape: vec![n],
            values: counts,
            dtype: DType::I64,
        })
    }

    /// One-dimensional linear interpolation (np.interp).
    /// `xp` and `fp` must be 1-D with the same length and `xp` sorted.
    pub fn interp(x: &Self, xp: &Self, fp: &Self) -> Result<Self, UFuncError> {
        if xp.shape.len() != 1 || fp.shape.len() != 1 {
            return Err(UFuncError::Msg("interp: xp and fp must be 1-D".to_string()));
        }
        if xp.shape[0] != fp.shape[0] {
            return Err(UFuncError::Msg(
                "interp: xp and fp must have same length".to_string(),
            ));
        }
        let n = xp.shape[0];
        if n == 0 {
            return Err(UFuncError::Msg("interp: xp must not be empty".to_string()));
        }

        let values: Vec<f64> = x
            .values
            .iter()
            .map(|&xi| {
                if n == 1 {
                    return fp.values[0];
                }
                if xi <= xp.values[0] {
                    return fp.values[0];
                }
                if xi >= xp.values[n - 1] {
                    return fp.values[n - 1];
                }
                // Binary search for interval
                let mut lo = 0;
                let mut hi = n - 1;
                while lo < hi - 1 {
                    let mid = (lo + hi) / 2;
                    if xp.values[mid] <= xi {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                let t = (xi - xp.values[lo]) / (xp.values[hi] - xp.values[lo]);
                fp.values[lo] * (1.0 - t) + fp.values[hi] * t
            })
            .collect();

        let shape = x.shape.clone();
        Ok(Self {
            shape,
            values,
            dtype: DType::F64,
        })
    }

    // ── statistics: quantile, average, cov, corrcoef, digitize, histogram2d ────

    /// Compute the q-th quantile (0.0–1.0 scale).
    ///
    /// Mimics `np.quantile(a, q)`. Like `percentile` but q is in [0, 1] instead of [0, 100].
    pub fn quantile(&self, q: f64, axis: Option<isize>) -> Result<Self, UFuncError> {
        self.percentile(q * 100.0, axis)
    }

    /// Weighted average of array elements.
    ///
    /// Mimics `np.average(a, weights=w)`. If `weights` is None, returns the mean.
    pub fn average(&self, weights: Option<&Self>, axis: Option<isize>) -> Result<Self, UFuncError> {
        match weights {
            None => self.reduce_mean(axis, false),
            Some(w) => {
                match axis {
                    None => {
                        if w.values.len() != self.values.len() {
                            return Err(UFuncError::InvalidInputLength {
                                expected: self.values.len(),
                                actual: w.values.len(),
                            });
                        }
                        let wsum: f64 = self
                            .values
                            .iter()
                            .zip(w.values.iter())
                            .map(|(&a, &b)| a * b)
                            .sum();
                        let wtot: f64 = w.values.iter().sum();
                        Ok(Self::scalar(
                            wsum / wtot,
                            promote_for_mean_reduction(self.dtype),
                        ))
                    }
                    Some(ax) => {
                        let ax = normalize_axis(ax, self.shape.len())?;
                        let axis_len = self.shape[ax];
                        if w.values.len() != axis_len {
                            return Err(UFuncError::InvalidInputLength {
                                expected: axis_len,
                                actual: w.values.len(),
                            });
                        }
                        // Multiply each lane element by its weight, sum, divide by weight total
                        let inner: usize = self.shape[ax + 1..].iter().copied().product();
                        let outer: usize = self.shape[..ax].iter().copied().product();
                        let wtot: f64 = w.values.iter().sum();
                        let out_shape = reduced_shape(&self.shape, ax, false);
                        let mut values = Vec::with_capacity(outer * inner);

                        for o in 0..outer {
                            for i in 0..inner {
                                let mut wsum = 0.0;
                                for k in 0..axis_len {
                                    let idx = o * axis_len * inner + k * inner + i;
                                    wsum += self.values[idx] * w.values[k];
                                }
                                values.push(wsum / wtot);
                            }
                        }
                        Ok(Self {
                            shape: out_shape,
                            values,
                            dtype: promote_for_mean_reduction(self.dtype),
                        })
                    }
                }
            }
        }
    }

    /// Estimate the covariance matrix.
    ///
    /// Mimics `np.cov(m)` where m is 2-D with each row as a variable.
    /// Returns the sample covariance matrix (ddof=1).
    pub fn cov(&self) -> Result<Self, UFuncError> {
        if self.shape.len() != 2 {
            return Err(UFuncError::Msg("cov: input must be 2-D".to_string()));
        }
        let (nvars, nobs) = (self.shape[0], self.shape[1]);
        if nobs < 2 {
            return Err(UFuncError::Msg(
                "cov: need at least 2 observations".to_string(),
            ));
        }
        // Compute means per row
        let means: Vec<f64> = (0..nvars)
            .map(|r| {
                let row = &self.values[r * nobs..(r + 1) * nobs];
                row.iter().sum::<f64>() / nobs as f64
            })
            .collect();

        // Compute covariance matrix
        let mut cov_values = vec![0.0; nvars * nvars];
        for i in 0..nvars {
            for j in i..nvars {
                let mut s = 0.0;
                for k in 0..nobs {
                    s += (self.values[i * nobs + k] - means[i])
                        * (self.values[j * nobs + k] - means[j]);
                }
                let c = s / (nobs - 1) as f64;
                cov_values[i * nvars + j] = c;
                cov_values[j * nvars + i] = c;
            }
        }
        Ok(Self {
            shape: vec![nvars, nvars],
            values: cov_values,
            dtype: DType::F64,
        })
    }

    /// Compute the Pearson correlation coefficient matrix.
    ///
    /// Mimics `np.corrcoef(x)` where x is 2-D with each row as a variable.
    pub fn corrcoef(&self) -> Result<Self, UFuncError> {
        let c = self.cov()?;
        let nvars = c.shape[0];
        let mut values = c.values.clone();
        // Normalize: corrcoef[i,j] = cov[i,j] / sqrt(cov[i,i] * cov[j,j])
        for i in 0..nvars {
            for j in 0..nvars {
                let denom = (c.values[i * nvars + i] * c.values[j * nvars + j]).sqrt();
                values[i * nvars + j] = if denom > 0.0 {
                    c.values[i * nvars + j] / denom
                } else {
                    f64::NAN
                };
            }
        }
        Ok(Self {
            shape: vec![nvars, nvars],
            values,
            dtype: DType::F64,
        })
    }

    /// Return the indices of the bins to which each value in the input belongs.
    ///
    /// Mimics `np.digitize(x, bins)`. Bins are assumed sorted ascending.
    /// Returns index i such that `bins[i-1] <= x < bins[i]`.
    pub fn digitize(&self, bins: &Self) -> Result<Self, UFuncError> {
        if bins.shape.len() != 1 {
            return Err(UFuncError::Msg("digitize: bins must be 1-D".to_string()));
        }
        let b = &bins.values;
        let values: Vec<f64> = self
            .values
            .iter()
            .map(|&v| {
                // Binary search: find rightmost bin index
                match b.binary_search_by(|probe| probe.total_cmp(&v)) {
                    Ok(idx) => (idx + 1) as f64,
                    Err(idx) => idx as f64,
                }
            })
            .collect();
        Ok(Self {
            shape: self.shape.clone(),
            values,
            dtype: DType::I64,
        })
    }

    /// Compute the 2-D histogram of two data samples.
    ///
    /// Mimics `np.histogram2d(x, y, bins)`.
    /// Returns `(hist, xedges, yedges)`.
    pub fn histogram2d(
        &self,
        y: &Self,
        xbins: usize,
        ybins: usize,
    ) -> Result<(Self, Self, Self), UFuncError> {
        if self.shape.len() != 1 || y.shape.len() != 1 {
            return Err(UFuncError::Msg(
                "histogram2d: x and y must be 1-D".to_string(),
            ));
        }
        if self.values.len() != y.values.len() {
            return Err(UFuncError::InvalidInputLength {
                expected: self.values.len(),
                actual: y.values.len(),
            });
        }
        let xmin = self.values.iter().copied().fold(f64::INFINITY, f64::min);
        let xmax = self
            .values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let ymin = y.values.iter().copied().fold(f64::INFINITY, f64::min);
        let ymax = y.values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let xedges = Self::linspace(xmin, xmax, xbins + 1, DType::F64)?;
        let yedges = Self::linspace(ymin, ymax, ybins + 1, DType::F64)?;

        let mut hist = vec![0.0f64; xbins * ybins];
        let xstep = if xbins > 0 && xmax > xmin {
            (xmax - xmin) / xbins as f64
        } else {
            1.0
        };
        let ystep = if ybins > 0 && ymax > ymin {
            (ymax - ymin) / ybins as f64
        } else {
            1.0
        };

        for (&xv, &yv) in self.values.iter().zip(y.values.iter()) {
            let xi = ((xv - xmin) / xstep).floor() as usize;
            let yi = ((yv - ymin) / ystep).floor() as usize;
            let xi = xi.min(xbins - 1);
            let yi = yi.min(ybins - 1);
            hist[xi * ybins + yi] += 1.0;
        }

        let h = Self {
            shape: vec![xbins, ybins],
            values: hist,
            dtype: DType::I64,
        };
        Ok((h, xedges, yedges))
    }

    // ── convolve, correlate, polyval, cross, vstack, hstack ────────

    /// Discrete linear convolution of two 1-D sequences (np.convolve, mode='full').
    pub fn convolve(&self, kernel: &Self) -> Result<Self, UFuncError> {
        if self.shape.len() != 1 || kernel.shape.len() != 1 {
            return Err(UFuncError::Msg(
                "convolve: only 1-D arrays supported".to_string(),
            ));
        }
        let n = self.shape[0];
        let m = kernel.shape[0];
        if n == 0 || m == 0 {
            return Ok(Self {
                shape: vec![0],
                values: Vec::new(),
                dtype: promote(self.dtype, kernel.dtype),
            });
        }
        let out_len = n + m - 1;
        let mut values = vec![0.0f64; out_len];
        for i in 0..n {
            for j in 0..m {
                values[i + j] += self.values[i] * kernel.values[j];
            }
        }
        Ok(Self {
            shape: vec![out_len],
            values,
            dtype: promote(self.dtype, kernel.dtype),
        })
    }

    /// Cross-correlation of two 1-D sequences (np.correlate, mode='full').
    pub fn correlate(&self, kernel: &Self) -> Result<Self, UFuncError> {
        if self.shape.len() != 1 || kernel.shape.len() != 1 {
            return Err(UFuncError::Msg(
                "correlate: only 1-D arrays supported".to_string(),
            ));
        }
        // correlate(a, v) = convolve(a, v[::-1])
        let reversed = Self {
            shape: kernel.shape.clone(),
            values: kernel.values.iter().rev().copied().collect(),
            dtype: kernel.dtype,
        };
        self.convolve(&reversed)
    }

    /// Evaluate a polynomial at given points (np.polyval).
    /// `coeffs` are in descending order: p(x) = c[0]*x^n + c[1]*x^(n-1) + ... + c[n].
    pub fn polyval(coeffs: &Self, x: &Self) -> Result<Self, UFuncError> {
        if coeffs.shape.len() != 1 {
            return Err(UFuncError::Msg(
                "polyval: coefficients must be 1-D".to_string(),
            ));
        }
        let values: Vec<f64> = x
            .values
            .iter()
            .map(|&xi| {
                // Horner's method
                let mut result = 0.0;
                for &c in &coeffs.values {
                    result = result * xi + c;
                }
                result
            })
            .collect();
        Ok(Self {
            shape: x.shape.clone(),
            values,
            dtype: DType::F64,
        })
    }

    /// Least-squares polynomial fit.
    ///
    /// Mimics `np.polyfit(x, y, deg)`. Returns coefficients in descending degree order.
    /// Uses the normal equations: (X^T X) c = X^T y.
    pub fn polyfit(x: &Self, y: &Self, deg: usize) -> Result<Self, UFuncError> {
        if x.shape.len() != 1 || y.shape.len() != 1 {
            return Err(UFuncError::Msg("polyfit: x and y must be 1-D".to_string()));
        }
        let n = x.values.len();
        if n != y.values.len() {
            return Err(UFuncError::InvalidInputLength {
                expected: n,
                actual: y.values.len(),
            });
        }
        if deg + 1 > n {
            return Err(UFuncError::Msg(
                "polyfit: degree too large for data".to_string(),
            ));
        }
        let m = deg + 1; // number of coefficients
        // Build Vandermonde matrix X: X[i][j] = x_i^(deg-j) (descending powers)
        // Then solve normal equations: (X^T X) c = X^T y
        // Build X^T X (m x m) and X^T y (m)
        let mut xtx = vec![0.0; m * m];
        let mut xty = vec![0.0; m];
        for i in 0..n {
            let xi = x.values[i];
            let yi = y.values[i];
            // Compute powers: xi^deg, xi^(deg-1), ..., xi^0
            let mut powers = vec![1.0f64; m];
            for j in (0..deg).rev() {
                powers[j] = powers[j + 1] * xi;
            }
            for r in 0..m {
                xty[r] += powers[r] * yi;
                for c in 0..m {
                    xtx[r * m + c] += powers[r] * powers[c];
                }
            }
        }
        // Solve via Gaussian elimination with partial pivoting
        let mut aug = vec![0.0; m * (m + 1)];
        for r in 0..m {
            for c in 0..m {
                aug[r * (m + 1) + c] = xtx[r * m + c];
            }
            aug[r * (m + 1) + m] = xty[r];
        }
        for col in 0..m {
            // Partial pivoting
            let mut max_row = col;
            let mut max_val = aug[col * (m + 1) + col].abs();
            for row in (col + 1)..m {
                let val = aug[row * (m + 1) + col].abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }
            if max_row != col {
                for c in 0..=m {
                    aug.swap(col * (m + 1) + c, max_row * (m + 1) + c);
                }
            }
            let pivot = aug[col * (m + 1) + col];
            if pivot.abs() < 1e-15 {
                return Err(UFuncError::Msg("polyfit: singular matrix".to_string()));
            }
            for c in col..=m {
                aug[col * (m + 1) + c] /= pivot;
            }
            for row in 0..m {
                if row == col {
                    continue;
                }
                let factor = aug[row * (m + 1) + col];
                for c in col..=m {
                    aug[row * (m + 1) + c] -= factor * aug[col * (m + 1) + c];
                }
            }
        }
        let coeffs: Vec<f64> = (0..m).map(|r| aug[r * (m + 1) + m]).collect();
        Ok(Self {
            shape: vec![m],
            values: coeffs,
            dtype: DType::F64,
        })
    }

    /// Compute the derivative of a polynomial.
    ///
    /// Mimics `np.polyder(p)`. Coefficients in descending degree order.
    pub fn polyder(&self) -> Result<Self, UFuncError> {
        if self.shape.len() != 1 {
            return Err(UFuncError::Msg(
                "polyder: coefficients must be 1-D".to_string(),
            ));
        }
        let n = self.values.len();
        if n <= 1 {
            return Ok(Self {
                shape: vec![1],
                values: vec![0.0],
                dtype: DType::F64,
            });
        }
        let deg = n - 1;
        let values: Vec<f64> = (0..deg)
            .map(|i| self.values[i] * (deg - i) as f64)
            .collect();
        Ok(Self {
            shape: vec![deg],
            values,
            dtype: DType::F64,
        })
    }

    /// Compute the antiderivative (integral) of a polynomial.
    ///
    /// Mimics `np.polyint(p)`. Coefficients in descending degree order.
    /// Integration constant is 0.
    pub fn polyint(&self) -> Result<Self, UFuncError> {
        if self.shape.len() != 1 {
            return Err(UFuncError::Msg(
                "polyint: coefficients must be 1-D".to_string(),
            ));
        }
        let n = self.values.len();
        let mut values = Vec::with_capacity(n + 1);
        for (i, &c) in self.values.iter().enumerate() {
            let power = n - i; // current degree
            values.push(c / power as f64);
        }
        values.push(0.0); // integration constant
        Ok(Self {
            shape: vec![n + 1],
            values,
            dtype: DType::F64,
        })
    }

    /// Find the roots of a polynomial using the companion matrix eigenvalue method.
    ///
    /// Mimics `np.roots(p)`. Only supports degree 1 and 2 polynomials analytically.
    /// Higher degrees return an error (full eigenvalue solver not yet implemented).
    pub fn roots(&self) -> Result<Self, UFuncError> {
        if self.shape.len() != 1 {
            return Err(UFuncError::Msg(
                "roots: coefficients must be 1-D".to_string(),
            ));
        }
        let n = self.values.len();
        if n <= 1 {
            return Ok(Self {
                shape: vec![0],
                values: Vec::new(),
                dtype: DType::F64,
            });
        }
        if n == 2 {
            // Linear: ax + b = 0 -> x = -b/a
            let root = -self.values[1] / self.values[0];
            return Ok(Self {
                shape: vec![1],
                values: vec![root],
                dtype: DType::F64,
            });
        }
        if n == 3 {
            // Quadratic: ax^2 + bx + c = 0
            let (a, b, c) = (self.values[0], self.values[1], self.values[2]);
            let disc = b * b - 4.0 * a * c;
            if disc < 0.0 {
                // Complex roots — return NaN for now (complex dtype not supported)
                return Ok(Self {
                    shape: vec![2],
                    values: vec![f64::NAN, f64::NAN],
                    dtype: DType::F64,
                });
            }
            let sqrt_disc = disc.sqrt();
            let r1 = (-b + sqrt_disc) / (2.0 * a);
            let r2 = (-b - sqrt_disc) / (2.0 * a);
            return Ok(Self {
                shape: vec![2],
                values: vec![r1, r2],
                dtype: DType::F64,
            });
        }
        Err(UFuncError::Msg(
            "roots: only degree 1 and 2 polynomials supported".to_string(),
        ))
    }

    // ── polynomial extensions ────────

    /// Construct polynomial coefficients from roots (np.poly).
    /// Returns coefficients in descending order with leading 1.
    pub fn poly(seq_of_zeros: &Self) -> Result<Self, UFuncError> {
        if seq_of_zeros.shape.len() != 1 {
            return Err(UFuncError::Msg("poly: input must be 1-D".to_string()));
        }
        // Start with [1.0] and multiply by (x - r) for each root
        let mut coeffs = vec![1.0];
        for &root in &seq_of_zeros.values {
            let mut new_coeffs = vec![0.0; coeffs.len() + 1];
            for (i, &c) in coeffs.iter().enumerate() {
                new_coeffs[i] += c;
                new_coeffs[i + 1] -= c * root;
            }
            coeffs = new_coeffs;
        }
        let n = coeffs.len();
        Ok(Self {
            shape: vec![n],
            values: coeffs,
            dtype: DType::F64,
        })
    }

    /// Multiply two polynomials (np.polymul). Coefficients in descending order.
    pub fn polymul(&self, other: &Self) -> Result<Self, UFuncError> {
        if self.shape.len() != 1 || other.shape.len() != 1 {
            return Err(UFuncError::Msg("polymul: inputs must be 1-D".to_string()));
        }
        let n = self.values.len();
        let m = other.values.len();
        if n == 0 || m == 0 {
            return Ok(Self {
                shape: vec![0],
                values: Vec::new(),
                dtype: DType::F64,
            });
        }
        let mut result = vec![0.0; n + m - 1];
        for (i, &a) in self.values.iter().enumerate() {
            for (j, &b) in other.values.iter().enumerate() {
                result[i + j] += a * b;
            }
        }
        let len = result.len();
        Ok(Self {
            shape: vec![len],
            values: result,
            dtype: DType::F64,
        })
    }

    /// Add two polynomials (np.polyadd). Coefficients in descending order.
    pub fn polyadd(&self, other: &Self) -> Result<Self, UFuncError> {
        if self.shape.len() != 1 || other.shape.len() != 1 {
            return Err(UFuncError::Msg("polyadd: inputs must be 1-D".to_string()));
        }
        let n = self.values.len().max(other.values.len());
        let mut result = vec![0.0; n];
        // Align from the right (descending order)
        for (i, &v) in self.values.iter().enumerate() {
            result[n - self.values.len() + i] += v;
        }
        for (i, &v) in other.values.iter().enumerate() {
            result[n - other.values.len() + i] += v;
        }
        Ok(Self {
            shape: vec![n],
            values: result,
            dtype: DType::F64,
        })
    }

    /// Subtract two polynomials (np.polysub). Coefficients in descending order.
    pub fn polysub(&self, other: &Self) -> Result<Self, UFuncError> {
        if self.shape.len() != 1 || other.shape.len() != 1 {
            return Err(UFuncError::Msg("polysub: inputs must be 1-D".to_string()));
        }
        let n = self.values.len().max(other.values.len());
        let mut result = vec![0.0; n];
        for (i, &v) in self.values.iter().enumerate() {
            result[n - self.values.len() + i] += v;
        }
        for (i, &v) in other.values.iter().enumerate() {
            result[n - other.values.len() + i] -= v;
        }
        Ok(Self {
            shape: vec![n],
            values: result,
            dtype: DType::F64,
        })
    }

    /// Divide two polynomials, returning (quotient, remainder) (np.polydiv).
    /// Coefficients in descending order.
    pub fn polydiv(&self, other: &Self) -> Result<(Self, Self), UFuncError> {
        if self.shape.len() != 1 || other.shape.len() != 1 {
            return Err(UFuncError::Msg("polydiv: inputs must be 1-D".to_string()));
        }
        if other.values.is_empty() || other.values.iter().all(|&v| v == 0.0) {
            return Err(UFuncError::Msg(
                "polydiv: division by zero polynomial".to_string(),
            ));
        }
        let mut remainder = self.values.clone();
        let divisor = &other.values;
        let n = remainder.len();
        let m = divisor.len();
        if n < m {
            return Ok((
                Self {
                    shape: vec![1],
                    values: vec![0.0],
                    dtype: DType::F64,
                },
                self.clone(),
            ));
        }
        let mut quotient = vec![0.0; n - m + 1];
        for i in 0..quotient.len() {
            let coeff = remainder[i] / divisor[0];
            quotient[i] = coeff;
            for (j, &d) in divisor.iter().enumerate() {
                remainder[i + j] -= coeff * d;
            }
        }
        // Trim leading zeros from remainder
        let rem_start = quotient.len();
        let rem_vals: Vec<f64> = remainder[rem_start..].to_vec();
        let rem_len = rem_vals.len().max(1);
        Ok((
            Self {
                shape: vec![quotient.len()],
                values: quotient,
                dtype: DType::F64,
            },
            Self {
                shape: vec![rem_len],
                values: if rem_vals.is_empty() {
                    vec![0.0]
                } else {
                    rem_vals
                },
                dtype: DType::F64,
            },
        ))
    }

    /// Cross product of two 3-element vectors (np.cross).
    pub fn cross(&self, rhs: &Self) -> Result<Self, UFuncError> {
        if self.shape.len() != 1 || rhs.shape.len() != 1 {
            return Err(UFuncError::Msg(
                "cross: only 1-D arrays supported".to_string(),
            ));
        }
        if self.shape[0] != 3 || rhs.shape[0] != 3 {
            return Err(UFuncError::Msg(
                "cross: inputs must have exactly 3 elements".to_string(),
            ));
        }
        let (a0, a1, a2) = (self.values[0], self.values[1], self.values[2]);
        let (b0, b1, b2) = (rhs.values[0], rhs.values[1], rhs.values[2]);
        let values = vec![a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0];
        let dtype = promote(self.dtype, rhs.dtype);
        Ok(Self {
            shape: vec![3],
            values,
            dtype,
        })
    }

    // ── window functions ────────

    /// Return the Hamming window of length M (np.hamming).
    pub fn hamming(m: usize) -> Self {
        if m <= 1 {
            return Self {
                shape: vec![m],
                values: vec![1.0; m],
                dtype: DType::F64,
            };
        }
        let values: Vec<f64> = (0..m)
            .map(|i| 0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (m as f64 - 1.0)).cos())
            .collect();
        Self {
            shape: vec![m],
            values,
            dtype: DType::F64,
        }
    }

    /// Return the Hanning window of length M (np.hanning).
    pub fn hanning(m: usize) -> Self {
        if m <= 1 {
            return Self {
                shape: vec![m],
                values: vec![1.0; m],
                dtype: DType::F64,
            };
        }
        let values: Vec<f64> = (0..m)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (m as f64 - 1.0)).cos())
            .collect();
        Self {
            shape: vec![m],
            values,
            dtype: DType::F64,
        }
    }

    /// Return the Blackman window of length M (np.blackman).
    pub fn blackman(m: usize) -> Self {
        if m <= 1 {
            return Self {
                shape: vec![m],
                values: vec![1.0; m],
                dtype: DType::F64,
            };
        }
        let values: Vec<f64> = (0..m)
            .map(|i| {
                let x = 2.0 * std::f64::consts::PI * i as f64 / (m as f64 - 1.0);
                0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos()
            })
            .collect();
        Self {
            shape: vec![m],
            values,
            dtype: DType::F64,
        }
    }

    /// Return the Bartlett (triangular) window of length M (np.bartlett).
    pub fn bartlett(m: usize) -> Self {
        if m <= 1 {
            return Self {
                shape: vec![m],
                values: vec![1.0; m],
                dtype: DType::F64,
            };
        }
        let half = (m as f64 - 1.0) / 2.0;
        let values: Vec<f64> = (0..m)
            .map(|i| 1.0 - ((i as f64 - half) / half).abs())
            .collect();
        Self {
            shape: vec![m],
            values,
            dtype: DType::F64,
        }
    }

    /// Return the Kaiser window of length M with shape parameter beta (np.kaiser).
    /// Uses a polynomial approximation of I0 (modified Bessel function of order 0).
    pub fn kaiser(m: usize, beta: f64) -> Self {
        if m <= 1 {
            return Self {
                shape: vec![m],
                values: vec![1.0; m],
                dtype: DType::F64,
            };
        }
        let values: Vec<f64> = (0..m)
            .map(|i| {
                let alpha = (m as f64 - 1.0) / 2.0;
                let r = (i as f64 - alpha) / alpha;
                let arg = beta * (1.0 - r * r).max(0.0).sqrt();
                bessel_i0(arg) / bessel_i0(beta)
            })
            .collect();
        Self {
            shape: vec![m],
            values,
            dtype: DType::F64,
        }
    }

    // ── FFT (Fast Fourier Transform) ────────────────────────────────

    /// Compute the 1-D discrete Fourier transform (np.fft.fft).
    /// Returns interleaved [re0, im0, re1, im1, ...] with shape [n].
    /// The `n` parameter specifies the length; input is zero-padded or truncated.
    /// If `n` is None, uses the length of the input array.
    pub fn fft(&self, n: Option<usize>) -> Result<Self, UFuncError> {
        let len = n.unwrap_or(self.values.len());
        if len == 0 {
            return Ok(Self {
                shape: vec![0],
                values: vec![],
                dtype: DType::F64,
            });
        }
        // Build complex input (real from self.values, imag = 0)
        let mut re = vec![0.0_f64; len];
        let mut im = vec![0.0_f64; len];
        for (i, v) in self.values.iter().enumerate() {
            if i >= len {
                break;
            }
            re[i] = *v;
        }
        fft_dit(&mut re, &mut im, false);
        // Interleave real/imag
        let mut values = Vec::with_capacity(len * 2);
        for i in 0..len {
            values.push(re[i]);
            values.push(im[i]);
        }
        Ok(Self {
            shape: vec![len, 2],
            values,
            dtype: DType::F64,
        })
    }

    /// Compute the 1-D inverse discrete Fourier transform (np.fft.ifft).
    /// Input is interleaved [re0, im0, re1, im1, ...] with shape [n, 2].
    /// Returns interleaved complex output with shape [n, 2].
    pub fn ifft(&self) -> Result<Self, UFuncError> {
        if self.shape.len() != 2 || self.shape[1] != 2 {
            return Err(UFuncError::Msg(
                "ifft: input must have shape [n, 2] (interleaved complex)".to_string(),
            ));
        }
        let len = self.shape[0];
        if len == 0 {
            return Ok(Self {
                shape: vec![0, 2],
                values: vec![],
                dtype: DType::F64,
            });
        }
        let mut re = Vec::with_capacity(len);
        let mut im = Vec::with_capacity(len);
        for i in 0..len {
            re.push(self.values[i * 2]);
            im.push(self.values[i * 2 + 1]);
        }
        fft_dit(&mut re, &mut im, true);
        let mut values = Vec::with_capacity(len * 2);
        for i in 0..len {
            values.push(re[i]);
            values.push(im[i]);
        }
        Ok(Self {
            shape: vec![len, 2],
            values,
            dtype: DType::F64,
        })
    }

    /// Compute the 1-D DFT for real input (np.fft.rfft).
    /// Returns the first n//2 + 1 complex coefficients as shape [n//2+1, 2].
    pub fn rfft(&self, n: Option<usize>) -> Result<Self, UFuncError> {
        let full = self.fft(n)?;
        let len = full.shape[0];
        let out_len = len / 2 + 1;
        let values = full.values[..out_len * 2].to_vec();
        Ok(Self {
            shape: vec![out_len, 2],
            values,
            dtype: DType::F64,
        })
    }

    /// Return the DFT sample frequencies (np.fft.fftfreq).
    /// For a signal of length `n` sampled at spacing `d`.
    pub fn fftfreq(n: usize, d: f64) -> Self {
        let val = 1.0 / (n as f64 * d);
        let half = n.div_ceil(2);
        let mut values: Vec<f64> = (0..half).map(|i| i as f64 * val).collect();
        let neg_start = -(n as i64 / 2);
        for idx in half..n {
            values.push((neg_start + (idx as i64 - half as i64)) as f64 * val);
        }
        Self {
            shape: vec![n],
            values,
            dtype: DType::F64,
        }
    }

    /// Return the DFT sample frequencies for rfft (np.fft.rfftfreq).
    pub fn rfftfreq(n: usize, d: f64) -> Self {
        let out_len = n / 2 + 1;
        let val = 1.0 / (n as f64 * d);
        let values: Vec<f64> = (0..out_len).map(|i| i as f64 * val).collect();
        Self {
            shape: vec![out_len],
            values,
            dtype: DType::F64,
        }
    }

    /// Shift the zero-frequency component to the center (np.fft.fftshift).
    /// Works on 1-D arrays.
    pub fn fftshift(&self) -> Result<Self, UFuncError> {
        if self.shape.len() != 1 {
            return Err(UFuncError::Msg(
                "fftshift: only 1-D arrays supported".to_string(),
            ));
        }
        let n = self.shape[0];
        let shift = n / 2;
        let values: Vec<f64> = (0..n).map(|i| self.values[(i + n - shift) % n]).collect();
        Ok(Self {
            shape: vec![n],
            values,
            dtype: DType::F64,
        })
    }

    /// Inverse of fftshift (np.fft.ifftshift).
    pub fn ifftshift(&self) -> Result<Self, UFuncError> {
        if self.shape.len() != 1 {
            return Err(UFuncError::Msg(
                "ifftshift: only 1-D arrays supported".to_string(),
            ));
        }
        let n = self.shape[0];
        let shift = n.div_ceil(2);
        let values: Vec<f64> = (0..n).map(|i| self.values[(i + n - shift) % n]).collect();
        Ok(Self {
            shape: vec![n],
            values,
            dtype: DType::F64,
        })
    }

    // ── Einsum ──────────────────────────────────────────────────────

    /// Einstein summation convention (np.einsum) for 1 or 2 operands.
    /// Supports subscript strings like "ij,jk->ik", "ij->", "i,i->", "ij->j".
    pub fn einsum(subscripts: &str, operands: &[&Self]) -> Result<Self, UFuncError> {
        let parts: Vec<&str> = subscripts.split("->").collect();
        if parts.len() != 2 {
            return Err(UFuncError::Msg(
                "einsum: subscripts must contain exactly one '->'".to_string(),
            ));
        }
        let input_part = parts[0];
        let output_labels: Vec<char> = parts[1].chars().collect();
        let input_subs: Vec<&str> = input_part.split(',').collect();

        if input_subs.len() != operands.len() {
            return Err(UFuncError::Msg(format!(
                "einsum: {} input subscripts but {} operands",
                input_subs.len(),
                operands.len()
            )));
        }
        if operands.is_empty() || operands.len() > 2 {
            return Err(UFuncError::Msg(
                "einsum: supports 1 or 2 operands".to_string(),
            ));
        }

        // Build label-to-dimension mapping
        let mut label_sizes: std::collections::HashMap<char, usize> =
            std::collections::HashMap::new();
        for (sub, op) in input_subs.iter().zip(operands.iter()) {
            let chars: Vec<char> = sub.chars().collect();
            if chars.len() != op.shape.len() {
                return Err(UFuncError::Msg(format!(
                    "einsum: subscript '{}' has {} indices but operand has {} dimensions",
                    sub,
                    chars.len(),
                    op.shape.len()
                )));
            }
            for (idx, &c) in chars.iter().enumerate() {
                let dim = op.shape[idx];
                if let Some(&existing) = label_sizes.get(&c) {
                    if existing != dim {
                        return Err(UFuncError::Msg(format!(
                            "einsum: conflicting sizes for label '{}': {} vs {}",
                            c, existing, dim
                        )));
                    }
                } else {
                    label_sizes.insert(c, dim);
                }
            }
        }

        // Determine output shape
        let output_shape: Vec<usize> = output_labels
            .iter()
            .map(|c| label_sizes.get(c).copied().unwrap_or(0))
            .collect();
        let output_size: usize = output_shape.iter().product::<usize>();

        // Determine contracted labels (in inputs but not in output)
        let mut all_input_labels: Vec<char> = Vec::new();
        for sub in &input_subs {
            for c in sub.chars() {
                if !all_input_labels.contains(&c) {
                    all_input_labels.push(c);
                }
            }
        }
        let contracted: Vec<char> = all_input_labels
            .iter()
            .filter(|c| !output_labels.contains(c))
            .copied()
            .collect();

        // All labels in iteration order: output labels + contracted labels
        let all_labels: Vec<char> = output_labels
            .iter()
            .chain(contracted.iter())
            .copied()
            .collect();
        let all_dims: Vec<usize> = all_labels
            .iter()
            .map(|c| label_sizes.get(c).copied().unwrap_or(1))
            .collect();
        let total_iters: usize = all_dims.iter().product::<usize>();

        let input_chars: Vec<Vec<char>> = input_subs.iter().map(|s| s.chars().collect()).collect();

        let mut result = vec![0.0; output_size];

        for flat_idx in 0..total_iters {
            // Decode flat index into label values
            let mut remaining = flat_idx;
            let mut label_vals: std::collections::HashMap<char, usize> =
                std::collections::HashMap::new();
            for i in (0..all_labels.len()).rev() {
                let dim = all_dims[i];
                label_vals.insert(all_labels[i], remaining % dim);
                remaining /= dim;
            }

            // Compute output flat index
            let mut out_flat = 0;
            let mut stride = 1;
            for i in (0..output_labels.len()).rev() {
                out_flat += label_vals[&output_labels[i]] * stride;
                stride *= output_shape[i];
            }

            // Compute product of operand elements
            let mut product = 1.0;
            for (op_idx, op) in operands.iter().enumerate() {
                let chars = &input_chars[op_idx];
                let mut op_flat = 0;
                let mut op_stride = 1;
                for i in (0..chars.len()).rev() {
                    op_flat += label_vals[&chars[i]] * op_stride;
                    op_stride *= op.shape[i];
                }
                product *= op.values[op_flat];
            }

            result[out_flat] += product;
        }

        Ok(Self {
            shape: output_shape,
            values: result,
            dtype: DType::F64,
        })
    }

    /// Stack arrays vertically (np.vstack). Row-wise concatenation.
    pub fn vstack(arrays: &[Self]) -> Result<Self, UFuncError> {
        if arrays.is_empty() {
            return Err(UFuncError::Msg("vstack: need at least 1 array".to_string()));
        }
        // Treat 1-D arrays as (1, N)
        let promoted: Vec<Self> = arrays
            .iter()
            .map(|a| {
                if a.shape.len() == 1 {
                    Self {
                        shape: vec![1, a.shape[0]],
                        values: a.values.clone(),
                        dtype: a.dtype,
                    }
                } else {
                    a.clone()
                }
            })
            .collect();
        Self::concatenate(&promoted.iter().collect::<Vec<_>>(), 0)
    }

    /// Stack arrays horizontally (np.hstack). Column-wise concatenation.
    pub fn hstack(arrays: &[Self]) -> Result<Self, UFuncError> {
        if arrays.is_empty() {
            return Err(UFuncError::Msg("hstack: need at least 1 array".to_string()));
        }
        // For 1-D arrays, concatenate along axis 0
        if arrays.iter().all(|a| a.shape.len() == 1) {
            return Self::concatenate(&arrays.iter().collect::<Vec<_>>(), 0);
        }
        Self::concatenate(&arrays.iter().collect::<Vec<_>>(), 1)
    }

    // ── slice, item, ravel, copy, fill, ptp, round, choose ────────

    /// Slice along an axis: equivalent to a[start:stop:step] along that axis.
    /// Supports negative indices. step must be positive.
    pub fn slice_axis(
        &self,
        axis: isize,
        start: Option<i64>,
        stop: Option<i64>,
        step: usize,
    ) -> Result<Self, UFuncError> {
        if step == 0 {
            return Err(UFuncError::Msg("slice: step cannot be 0".to_string()));
        }
        let ax = normalize_axis(axis, self.shape.len())?;
        let axis_len = self.shape[ax] as i64;

        let resolve = |val: i64| -> i64 {
            if val < 0 {
                (val + axis_len).max(0)
            } else {
                val.min(axis_len)
            }
        };
        let s = resolve(start.unwrap_or(0));
        let e = resolve(stop.unwrap_or(axis_len));
        if s >= e {
            let mut out_shape = self.shape.clone();
            out_shape[ax] = 0;
            return Ok(Self {
                shape: out_shape,
                values: Vec::new(),
                dtype: self.dtype,
            });
        }

        let indices: Vec<i64> = (s..e).step_by(step).collect();
        self.take(&indices, Some(ax as isize))
    }

    /// Get a single element by multi-dimensional index. Returns a scalar.
    pub fn item(&self, index: &[i64]) -> Result<f64, UFuncError> {
        if index.len() != self.shape.len() {
            return Err(UFuncError::Msg(format!(
                "item: index has {} dims but array has {}",
                index.len(),
                self.shape.len()
            )));
        }
        let mut flat = 0usize;
        let strides = c_strides_elems(&self.shape);
        for (i, (&idx, &dim)) in index.iter().zip(&self.shape).enumerate() {
            let resolved = if idx < 0 { idx + dim as i64 } else { idx };
            if resolved < 0 || resolved >= dim as i64 {
                return Err(UFuncError::Msg(format!(
                    "item: index {idx} out of bounds for axis {i} with size {dim}"
                )));
            }
            flat += resolved as usize * strides[i];
        }
        Ok(self.values[flat])
    }

    /// Set a single element by multi-dimensional index.
    pub fn itemset(&mut self, index: &[i64], value: f64) -> Result<(), UFuncError> {
        if index.len() != self.shape.len() {
            return Err(UFuncError::Msg(format!(
                "itemset: index has {} dims but array has {}",
                index.len(),
                self.shape.len()
            )));
        }
        let mut flat = 0usize;
        let strides = c_strides_elems(&self.shape);
        for (i, (&idx, &dim)) in index.iter().zip(&self.shape).enumerate() {
            let resolved = if idx < 0 { idx + dim as i64 } else { idx };
            if resolved < 0 || resolved >= dim as i64 {
                return Err(UFuncError::Msg(format!(
                    "itemset: index {idx} out of bounds for axis {i} with size {dim}"
                )));
            }
            flat += resolved as usize * strides[i];
        }
        self.values[flat] = value;
        Ok(())
    }

    /// Return a contiguous flattened copy (np.ravel).
    pub fn ravel(&self) -> Self {
        let n = self.values.len();
        Self {
            shape: vec![n],
            values: self.values.clone(),
            dtype: self.dtype,
        }
    }

    /// Return a copy of the array.
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Fill the array with a scalar value in-place.
    pub fn fill(&mut self, value: f64) {
        for v in &mut self.values {
            *v = value;
        }
    }

    /// Peak-to-peak (max - min) along an axis or over all elements.
    pub fn ptp(&self, axis: Option<isize>) -> Result<Self, UFuncError> {
        match axis {
            None => {
                if self.values.is_empty() {
                    return Err(UFuncError::Msg("ptp of empty array".to_string()));
                }
                let min = self.values.iter().copied().fold(f64::INFINITY, f64::min);
                let max = self
                    .values
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                Ok(Self::scalar(max - min, self.dtype))
            }
            Some(ax) => {
                let mx = self.reduce_max(Some(ax), false)?;
                let mn = self.reduce_min(Some(ax), false)?;
                mx.elementwise_binary(&mn, BinaryOp::Sub)
            }
        }
    }

    /// Round to the given number of decimal places (np.around).
    pub fn round_to(&self, decimals: i32) -> Self {
        let factor = 10.0f64.powi(decimals);
        let values: Vec<f64> = self
            .values
            .iter()
            .map(|&v| (v * factor).round() / factor)
            .collect();
        Self {
            shape: self.shape.clone(),
            values,
            dtype: self.dtype,
        }
    }

    /// Choose elements from a list of arrays based on an index array (np.choose).
    /// `self` contains integer indices selecting which array to pick from.
    pub fn choose(&self, choices: &[Self]) -> Result<Self, UFuncError> {
        if choices.is_empty() {
            return Err(UFuncError::Msg(
                "choose: need at least 1 choice".to_string(),
            ));
        }
        let n_choices = choices.len();
        let mut values = Vec::with_capacity(self.values.len());
        for (i, &idx) in self.values.iter().enumerate() {
            let choice_idx = idx as usize;
            if choice_idx >= n_choices {
                return Err(UFuncError::Msg(format!(
                    "choose: index {choice_idx} out of range for {n_choices} choices"
                )));
            }
            if i >= choices[choice_idx].values.len() {
                return Err(UFuncError::Msg(format!(
                    "choose: position {i} out of range for choice {choice_idx}"
                )));
            }
            values.push(choices[choice_idx].values[i]);
        }
        let dtype = choices[0].dtype;
        Ok(Self {
            shape: self.shape.clone(),
            values,
            dtype,
        })
    }

    /// Return sorted unique elements.
    pub fn unique(&self) -> Self {
        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        sorted.dedup_by(|a, b| a.total_cmp(b).is_eq());
        let n = sorted.len();
        Self {
            shape: vec![n],
            values: sorted,
            dtype: self.dtype,
        }
    }

    /// Return sorted unique elements along with optional index/inverse/counts arrays.
    ///
    /// Mimics `np.unique(return_index, return_inverse, return_counts)`.
    /// Returns `(unique, Option<indices>, Option<inverse>, Option<counts>)`.
    pub fn unique_with_info(
        &self,
        return_index: bool,
        return_inverse: bool,
        return_counts: bool,
    ) -> (Self, Option<Self>, Option<Self>, Option<Self>) {
        let flat = &self.values;
        // Build (value, original_index) pairs, sort by value.
        let mut indexed: Vec<(f64, usize)> = flat
            .iter()
            .copied()
            .enumerate()
            .map(|(i, v)| (v, i))
            .collect();
        indexed.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut unique_vals: Vec<f64> = Vec::new();
        let mut first_indices: Vec<f64> = Vec::new();
        let mut counts_vec: Vec<f64> = Vec::new();
        // Map from sorted position → unique group index (used to build inverse)
        let mut group_of_sorted: Vec<usize> = vec![0; flat.len()];

        let mut group = 0usize;
        for (i, &(val, orig_idx)) in indexed.iter().enumerate() {
            if i == 0 || val.total_cmp(&indexed[i - 1].0) != std::cmp::Ordering::Equal {
                unique_vals.push(val);
                first_indices.push(orig_idx as f64);
                counts_vec.push(1.0);
                if i > 0 {
                    group += 1;
                }
            } else {
                *counts_vec.last_mut().unwrap() += 1.0;
            }
            group_of_sorted[i] = group;
        }

        // Build inverse: for each original index, which unique group does it belong to?
        let inverse_vec: Vec<f64> = if return_inverse {
            let mut inv = vec![0.0; flat.len()];
            for (sorted_pos, &(_, orig_idx)) in indexed.iter().enumerate() {
                inv[orig_idx] = group_of_sorted[sorted_pos] as f64;
            }
            inv
        } else {
            Vec::new()
        };

        let nu = unique_vals.len();
        let n = flat.len();
        let unique = Self {
            shape: vec![nu],
            values: unique_vals,
            dtype: self.dtype,
        };
        let indices = if return_index {
            Some(Self {
                shape: vec![nu],
                values: first_indices,
                dtype: DType::I64,
            })
        } else {
            None
        };
        let inverse = if return_inverse {
            Some(Self {
                shape: vec![n],
                values: inverse_vec,
                dtype: DType::I64,
            })
        } else {
            None
        };
        let counts = if return_counts {
            Some(Self {
                shape: vec![nu],
                values: counts_vec,
                dtype: DType::I64,
            })
        } else {
            None
        };
        (unique, indices, inverse, counts)
    }

    /// Test whether each element of `self` is in `test_elements`. Returns a boolean array.
    ///
    /// Mimics `np.in1d(ar1, ar2)`.
    pub fn in1d(&self, test_elements: &Self) -> Self {
        let mut set: Vec<f64> = test_elements.values.clone();
        set.sort_by(|a, b| a.total_cmp(b));
        set.dedup_by(|a, b| a.total_cmp(b).is_eq());
        let values: Vec<f64> = self
            .values
            .iter()
            .map(|&v| {
                if set.binary_search_by(|x| x.total_cmp(&v)).is_ok() {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        Self {
            shape: self.shape.clone(),
            values,
            dtype: DType::Bool,
        }
    }

    /// Test whether each element of `self` is in `test_elements`. Returns a boolean array.
    ///
    /// Alias matching `np.isin(element, test_elements)` — same as `in1d` but preserves shape.
    pub fn isin(&self, test_elements: &Self) -> Self {
        self.in1d(test_elements)
    }

    /// Return the sorted union of two 1-D arrays.
    ///
    /// Mimics `np.union1d(ar1, ar2)`.
    pub fn union1d(&self, other: &Self) -> Self {
        let mut combined: Vec<f64> = self
            .values
            .iter()
            .chain(other.values.iter())
            .copied()
            .collect();
        combined.sort_by(|a, b| a.total_cmp(b));
        combined.dedup_by(|a, b| a.total_cmp(b).is_eq());
        let n = combined.len();
        Self {
            shape: vec![n],
            values: combined,
            dtype: self.dtype,
        }
    }

    /// Return the sorted intersection of two 1-D arrays.
    ///
    /// Mimics `np.intersect1d(ar1, ar2)`.
    pub fn intersect1d(&self, other: &Self) -> Self {
        let mut a = self.values.clone();
        a.sort_by(|x, y| x.total_cmp(y));
        a.dedup_by(|x, y| x.total_cmp(y).is_eq());
        let mut b = other.values.clone();
        b.sort_by(|x, y| x.total_cmp(y));
        b.dedup_by(|x, y| x.total_cmp(y).is_eq());

        let mut result = Vec::new();
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            match a[i].total_cmp(&b[j]) {
                std::cmp::Ordering::Equal => {
                    result.push(a[i]);
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        let n = result.len();
        Self {
            shape: vec![n],
            values: result,
            dtype: self.dtype,
        }
    }

    /// Return the sorted set difference of two 1-D arrays (elements in self not in other).
    ///
    /// Mimics `np.setdiff1d(ar1, ar2)`.
    pub fn setdiff1d(&self, other: &Self) -> Self {
        let mut b = other.values.clone();
        b.sort_by(|x, y| x.total_cmp(y));
        b.dedup_by(|x, y| x.total_cmp(y).is_eq());

        let mut a = self.values.clone();
        a.sort_by(|x, y| x.total_cmp(y));
        a.dedup_by(|x, y| x.total_cmp(y).is_eq());

        let result: Vec<f64> = a
            .into_iter()
            .filter(|v| b.binary_search_by(|x| x.total_cmp(v)).is_err())
            .collect();
        let n = result.len();
        Self {
            shape: vec![n],
            values: result,
            dtype: self.dtype,
        }
    }

    /// Return the sorted symmetric difference of two 1-D arrays.
    ///
    /// Mimics `np.setxor1d(ar1, ar2)`.
    pub fn setxor1d(&self, other: &Self) -> Self {
        let mut a = self.values.clone();
        a.sort_by(|x, y| x.total_cmp(y));
        a.dedup_by(|x, y| x.total_cmp(y).is_eq());
        let mut b = other.values.clone();
        b.sort_by(|x, y| x.total_cmp(y));
        b.dedup_by(|x, y| x.total_cmp(y).is_eq());

        // Elements in a but not b, plus elements in b but not a
        let mut result: Vec<f64> = a
            .iter()
            .filter(|v| b.binary_search_by(|x| x.total_cmp(v)).is_err())
            .chain(
                b.iter()
                    .filter(|v| a.binary_search_by(|x| x.total_cmp(v)).is_err()),
            )
            .copied()
            .collect();
        result.sort_by(|x, y| x.total_cmp(y));
        let n = result.len();
        Self {
            shape: vec![n],
            values: result,
            dtype: self.dtype,
        }
    }

    // ── NaN-aware reductions ────────────────────────────────────────────

    /// Helper: produce a copy of `self` with NaN values removed (flattened).
    fn nan_filtered(&self) -> Self {
        let values: Vec<f64> = self
            .values
            .iter()
            .copied()
            .filter(|v| !v.is_nan())
            .collect();
        let n = values.len();
        Self {
            shape: vec![n],
            values,
            dtype: self.dtype,
        }
    }

    /// Helper: produce a copy with NaN values removed along a specific axis
    /// by replacing NaN with `fill` before the reduction.
    /// Returns (filled array, per-lane NaN counts).
    fn nan_fill_for_axis(&self, axis: usize, fill: f64) -> (Self, Vec<usize>) {
        let strides = c_strides_elems(&self.shape);
        let axis_len = self.shape[axis];
        let outer_count = self.values.len() / axis_len;

        let mut filled = self.values.clone();
        let mut nan_counts = vec![0usize; outer_count];

        for (outer, nan_count) in nan_counts.iter_mut().enumerate() {
            // Map outer index to multi-index skipping the reduction axis
            let mut remainder = outer;
            let mut base_flat = 0usize;
            for (d, (&_s, &stride)) in self.shape.iter().zip(strides.iter()).enumerate() {
                if d == axis {
                    continue;
                }
                let outer_stride = if d < axis {
                    strides[d] / axis_len
                } else {
                    strides[d]
                };
                let coord = remainder / outer_stride;
                remainder %= outer_stride;
                base_flat += coord * stride;
            }
            for k in 0..axis_len {
                let idx = base_flat + k * strides[axis];
                if filled[idx].is_nan() {
                    filled[idx] = fill;
                    *nan_count += 1;
                }
            }
        }

        let arr = Self {
            shape: self.shape.clone(),
            values: filled,
            dtype: self.dtype,
        };
        (arr, nan_counts)
    }

    /// `np.nansum` — sum ignoring NaN values.
    pub fn nansum(&self, axis: Option<isize>, keepdims: bool) -> Result<Self, UFuncError> {
        match axis {
            None => {
                let sum: f64 = self.values.iter().copied().filter(|v| !v.is_nan()).sum();
                let shape = if keepdims {
                    vec![1; self.shape.len()]
                } else {
                    Vec::new()
                };
                Ok(Self {
                    shape,
                    values: vec![sum],
                    dtype: promote_for_sum_reduction(self.dtype),
                })
            }
            Some(ax) => {
                let ax = normalize_axis(ax, self.shape.len())?;
                let (filled, _) = self.nan_fill_for_axis(ax, 0.0);
                filled.reduce_sum(Some(ax as isize), keepdims)
            }
        }
    }

    /// `np.nanmean` — mean ignoring NaN values.
    pub fn nanmean(&self, axis: Option<isize>, keepdims: bool) -> Result<Self, UFuncError> {
        match axis {
            None => {
                let non_nan: Vec<f64> = self
                    .values
                    .iter()
                    .copied()
                    .filter(|v| !v.is_nan())
                    .collect();
                let n = non_nan.len() as f64;
                let sum: f64 = non_nan.iter().sum();
                let shape = if keepdims {
                    vec![1; self.shape.len()]
                } else {
                    Vec::new()
                };
                Ok(Self {
                    shape,
                    values: vec![sum / n],
                    dtype: promote_for_mean_reduction(self.dtype),
                })
            }
            Some(ax) => {
                let ax = normalize_axis(ax, self.shape.len())?;
                let axis_len = self.shape[ax];
                let (filled, nan_counts) = self.nan_fill_for_axis(ax, 0.0);
                let mut result = filled.reduce_sum(Some(ax as isize), keepdims)?;
                for (i, val) in result.values.iter_mut().enumerate() {
                    let valid = (axis_len - nan_counts[i]) as f64;
                    *val /= valid;
                }
                result.dtype = promote_for_mean_reduction(self.dtype);
                Ok(result)
            }
        }
    }

    /// `np.nanvar` — variance ignoring NaN values (ddof=0).
    pub fn nanvar(
        &self,
        axis: Option<isize>,
        keepdims: bool,
        ddof: usize,
    ) -> Result<Self, UFuncError> {
        match axis {
            None => {
                let non_nan: Vec<f64> = self
                    .values
                    .iter()
                    .copied()
                    .filter(|v| !v.is_nan())
                    .collect();
                let n = non_nan.len();
                let mean = non_nan.iter().sum::<f64>() / n as f64;
                let var =
                    non_nan.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - ddof) as f64;
                let shape = if keepdims {
                    vec![1; self.shape.len()]
                } else {
                    Vec::new()
                };
                Ok(Self {
                    shape,
                    values: vec![var],
                    dtype: promote_for_mean_reduction(self.dtype),
                })
            }
            Some(ax) => {
                let ax = normalize_axis(ax, self.shape.len())?;
                // Compute nanmean first, then compute variance from filled data
                let mean_arr = self.nanmean(Some(ax as isize), true)?;
                let axis_len = self.shape[ax];
                let strides = c_strides_elems(&self.shape);

                let out_shape = reduced_shape(&self.shape, ax, keepdims);
                let out_count = element_count(&out_shape).map_err(UFuncError::Shape)?;
                let mut out_values = vec![0.0f64; out_count];

                for (outer, out_val) in out_values.iter_mut().enumerate() {
                    let mean_val = mean_arr.values[outer];
                    let mut sq_sum = 0.0;
                    let mut valid = 0usize;
                    let mut remainder = outer;
                    let mut base_flat = 0usize;
                    for (d, (&_s, &stride)) in self.shape.iter().zip(strides.iter()).enumerate() {
                        if d == ax {
                            continue;
                        }
                        let outer_stride = if d < ax {
                            strides[d] / axis_len
                        } else {
                            strides[d]
                        };
                        let coord = remainder / outer_stride;
                        remainder %= outer_stride;
                        base_flat += coord * stride;
                    }
                    for k in 0..axis_len {
                        let v = self.values[base_flat + k * strides[ax]];
                        if !v.is_nan() {
                            sq_sum += (v - mean_val).powi(2);
                            valid += 1;
                        }
                    }
                    *out_val = sq_sum / (valid - ddof) as f64;
                }

                Ok(Self {
                    shape: out_shape,
                    values: out_values,
                    dtype: promote_for_mean_reduction(self.dtype),
                })
            }
        }
    }

    /// `np.nanstd` — standard deviation ignoring NaN values (ddof=0).
    pub fn nanstd(
        &self,
        axis: Option<isize>,
        keepdims: bool,
        ddof: usize,
    ) -> Result<Self, UFuncError> {
        let mut var = self.nanvar(axis, keepdims, ddof)?;
        for v in &mut var.values {
            *v = v.sqrt();
        }
        Ok(var)
    }

    /// `np.nanmin` — minimum ignoring NaN values.
    pub fn nanmin(&self, axis: Option<isize>, keepdims: bool) -> Result<Self, UFuncError> {
        match axis {
            None => {
                let min = self
                    .values
                    .iter()
                    .copied()
                    .filter(|v| !v.is_nan())
                    .fold(f64::INFINITY, f64::min);
                let shape = if keepdims {
                    vec![1; self.shape.len()]
                } else {
                    Vec::new()
                };
                Ok(Self {
                    shape,
                    values: vec![min],
                    dtype: self.dtype,
                })
            }
            Some(ax) => {
                let ax = normalize_axis(ax, self.shape.len())?;
                let (filled, _) = self.nan_fill_for_axis(ax, f64::INFINITY);
                filled.reduce_min(Some(ax as isize), keepdims)
            }
        }
    }

    /// `np.nanmax` — maximum ignoring NaN values.
    pub fn nanmax(&self, axis: Option<isize>, keepdims: bool) -> Result<Self, UFuncError> {
        match axis {
            None => {
                let max = self
                    .values
                    .iter()
                    .copied()
                    .filter(|v| !v.is_nan())
                    .fold(f64::NEG_INFINITY, f64::max);
                let shape = if keepdims {
                    vec![1; self.shape.len()]
                } else {
                    Vec::new()
                };
                Ok(Self {
                    shape,
                    values: vec![max],
                    dtype: self.dtype,
                })
            }
            Some(ax) => {
                let ax = normalize_axis(ax, self.shape.len())?;
                let (filled, _) = self.nan_fill_for_axis(ax, f64::NEG_INFINITY);
                filled.reduce_max(Some(ax as isize), keepdims)
            }
        }
    }

    /// `np.nanmedian` — median ignoring NaN values.
    pub fn nanmedian(&self, axis: Option<isize>) -> Result<Self, UFuncError> {
        match axis {
            None => {
                let filtered = self.nan_filtered();
                filtered.median(None)
            }
            Some(ax) => {
                let ax = normalize_axis(ax, self.shape.len())?;
                let axis_len = self.shape[ax];
                let strides = c_strides_elems(&self.shape);
                let out_shape = reduced_shape(&self.shape, ax, false);
                let outer_count = self.values.len() / axis_len;
                let mut out_values = Vec::with_capacity(outer_count);

                for outer in 0..outer_count {
                    let mut remainder = outer;
                    let mut base_flat = 0usize;
                    for (d, (&_s, &stride)) in self.shape.iter().zip(strides.iter()).enumerate() {
                        if d == ax {
                            continue;
                        }
                        let outer_stride = if d < ax {
                            strides[d] / axis_len
                        } else {
                            strides[d]
                        };
                        let coord = remainder / outer_stride;
                        remainder %= outer_stride;
                        base_flat += coord * stride;
                    }
                    let mut lane: Vec<f64> = (0..axis_len)
                        .map(|k| self.values[base_flat + k * strides[ax]])
                        .filter(|v| !v.is_nan())
                        .collect();
                    lane.sort_by(|a, b| a.total_cmp(b));
                    let median = interpolate_percentile(&lane, 0.5);
                    out_values.push(median);
                }
                Ok(Self {
                    shape: out_shape,
                    values: out_values,
                    dtype: promote_for_mean_reduction(self.dtype),
                })
            }
        }
    }

    /// `np.nanargmin` — index of minimum ignoring NaN values.
    pub fn nanargmin(&self, axis: Option<isize>) -> Result<Self, UFuncError> {
        match axis {
            None => {
                let mut min_idx = None;
                let mut min_val = f64::INFINITY;
                for (i, &v) in self.values.iter().enumerate() {
                    if !v.is_nan() && v < min_val {
                        min_val = v;
                        min_idx = Some(i);
                    }
                }
                match min_idx {
                    Some(idx) => Ok(Self::scalar(idx as f64, DType::I64)),
                    None => Err(UFuncError::Msg(
                        "nanargmin: All-NaN slice encountered".to_string(),
                    )),
                }
            }
            Some(ax) => {
                let ax = normalize_axis(ax, self.shape.len())?;
                let axis_len = self.shape[ax];
                if axis_len == 0 {
                    return Err(UFuncError::Msg(
                        "nanargmin: zero-size array to reduction operation".to_string(),
                    ));
                }
                let strides = c_strides_elems(&self.shape);
                let out_shape = reduced_shape(&self.shape, ax, false);
                let outer_count = self.values.len() / axis_len;
                let mut out_values = Vec::with_capacity(outer_count);

                for outer in 0..outer_count {
                    let mut remainder = outer;
                    let mut base_flat = 0usize;
                    for (d, (&_s, &stride)) in self.shape.iter().zip(strides.iter()).enumerate() {
                        if d == ax {
                            continue;
                        }
                        let outer_stride = if d < ax {
                            strides[d] / axis_len
                        } else {
                            strides[d]
                        };
                        let coord = remainder / outer_stride;
                        remainder %= outer_stride;
                        base_flat += coord * stride;
                    }
                    let mut best_k = None;
                    let mut best_v = f64::INFINITY;
                    for k in 0..axis_len {
                        let v = self.values[base_flat + k * strides[ax]];
                        if !v.is_nan() && v < best_v {
                            best_v = v;
                            best_k = Some(k);
                        }
                    }
                    match best_k {
                        Some(k) => out_values.push(k as f64),
                        None => {
                            return Err(UFuncError::Msg(
                                "nanargmin: All-NaN slice encountered".to_string(),
                            ));
                        }
                    }
                }
                Ok(Self {
                    shape: out_shape,
                    values: out_values,
                    dtype: DType::I64,
                })
            }
        }
    }

    /// `np.nanargmax` — index of maximum ignoring NaN values.
    pub fn nanargmax(&self, axis: Option<isize>) -> Result<Self, UFuncError> {
        match axis {
            None => {
                let mut max_idx = None;
                let mut max_val = f64::NEG_INFINITY;
                for (i, &v) in self.values.iter().enumerate() {
                    if !v.is_nan() && v > max_val {
                        max_val = v;
                        max_idx = Some(i);
                    }
                }
                match max_idx {
                    Some(idx) => Ok(Self::scalar(idx as f64, DType::I64)),
                    None => Err(UFuncError::Msg(
                        "nanargmax: All-NaN slice encountered".to_string(),
                    )),
                }
            }
            Some(ax) => {
                let ax = normalize_axis(ax, self.shape.len())?;
                let axis_len = self.shape[ax];
                if axis_len == 0 {
                    return Err(UFuncError::Msg(
                        "nanargmax: zero-size array to reduction operation".to_string(),
                    ));
                }
                let strides = c_strides_elems(&self.shape);
                let out_shape = reduced_shape(&self.shape, ax, false);
                let outer_count = self.values.len() / axis_len;
                let mut out_values = Vec::with_capacity(outer_count);

                for outer in 0..outer_count {
                    let mut remainder = outer;
                    let mut base_flat = 0usize;
                    for (d, (&_s, &stride)) in self.shape.iter().zip(strides.iter()).enumerate() {
                        if d == ax {
                            continue;
                        }
                        let outer_stride = if d < ax {
                            strides[d] / axis_len
                        } else {
                            strides[d]
                        };
                        let coord = remainder / outer_stride;
                        remainder %= outer_stride;
                        base_flat += coord * stride;
                    }
                    let mut best_k = None;
                    let mut best_v = f64::NEG_INFINITY;
                    for k in 0..axis_len {
                        let v = self.values[base_flat + k * strides[ax]];
                        if !v.is_nan() && v > best_v {
                            best_v = v;
                            best_k = Some(k);
                        }
                    }
                    match best_k {
                        Some(k) => out_values.push(k as f64),
                        None => {
                            return Err(UFuncError::Msg(
                                "nanargmax: All-NaN slice encountered".to_string(),
                            ));
                        }
                    }
                }
                Ok(Self {
                    shape: out_shape,
                    values: out_values,
                    dtype: DType::I64,
                })
            }
        }
    }

    /// `np.nanprod` — product of elements ignoring NaN values (treated as 1).
    pub fn nanprod(&self, axis: Option<isize>, keepdims: bool) -> Result<Self, UFuncError> {
        let filled = self.nan_filtered_fill(1.0);
        filled.reduce_prod(axis, keepdims)
    }

    /// `np.nancumsum` — cumulative sum ignoring NaN (treated as 0).
    pub fn nancumsum(&self, axis: Option<isize>) -> Result<Self, UFuncError> {
        let filled = self.nan_filtered_fill(0.0);
        filled.cumsum(axis)
    }

    /// `np.nancumprod` — cumulative product ignoring NaN (treated as 1).
    pub fn nancumprod(&self, axis: Option<isize>) -> Result<Self, UFuncError> {
        let filled = self.nan_filtered_fill(1.0);
        filled.cumprod(axis)
    }

    /// Replace NaN values with a specified fill value.
    fn nan_filtered_fill(&self, fill: f64) -> Self {
        let values: Vec<f64> = self
            .values
            .iter()
            .map(|&v| if v.is_nan() { fill } else { v })
            .collect();
        Self {
            shape: self.shape.clone(),
            values,
            dtype: self.dtype,
        }
    }

    // ── misc math ────────

    /// Vandermonde matrix (np.vander).
    /// Returns a matrix where columns are powers of the input vector.
    pub fn vander(&self, n: Option<usize>, increasing: bool) -> Result<Self, UFuncError> {
        if self.shape.len() != 1 {
            return Err(UFuncError::Msg("vander: input must be 1-D".to_string()));
        }
        let m = self.shape[0];
        let cols = n.unwrap_or(m);
        let mut values = Vec::with_capacity(m * cols);
        for &x in &self.values {
            if increasing {
                for j in 0..cols {
                    values.push(x.powi(j as i32));
                }
            } else {
                for j in (0..cols).rev() {
                    values.push(x.powi(j as i32));
                }
            }
        }
        Ok(Self {
            shape: vec![m, cols],
            values,
            dtype: DType::F64,
        })
    }

    /// Differences between consecutive elements of an array (np.ediff1d).
    pub fn ediff1d(&self) -> Result<Self, UFuncError> {
        let flat = &self.values;
        if flat.len() < 2 {
            return Ok(Self {
                shape: vec![0],
                values: Vec::new(),
                dtype: self.dtype,
            });
        }
        let values: Vec<f64> = flat.windows(2).map(|w| w[1] - w[0]).collect();
        let n = values.len();
        Ok(Self {
            shape: vec![n],
            values,
            dtype: self.dtype,
        })
    }

    /// Trapezoidal integration (np.trapezoid / np.trapz).
    pub fn trapezoid(&self, dx: f64) -> Result<Self, UFuncError> {
        if self.shape.len() != 1 {
            return Err(UFuncError::Msg(
                "trapezoid: only 1-D arrays supported".to_string(),
            ));
        }
        if self.values.len() < 2 {
            return Ok(Self::scalar(0.0, self.dtype));
        }
        let sum: f64 = self
            .values
            .windows(2)
            .map(|w| (w[0] + w[1]) / 2.0 * dx)
            .sum();
        Ok(Self::scalar(sum, DType::F64))
    }

    /// Sinc function (np.sinc): sin(pi*x) / (pi*x), with sinc(0)=1.
    pub fn sinc(&self) -> Self {
        let values: Vec<f64> = self
            .values
            .iter()
            .map(|&x| {
                if x == 0.0 {
                    1.0
                } else {
                    let px = std::f64::consts::PI * x;
                    px.sin() / px
                }
            })
            .collect();
        Self {
            shape: self.shape.clone(),
            values,
            dtype: DType::F64,
        }
    }

    /// Modified Bessel function of the first kind, order 0 (np.i0).
    pub fn i0(&self) -> Self {
        let values: Vec<f64> = self.values.iter().map(|&x| bessel_i0(x)).collect();
        Self {
            shape: self.shape.clone(),
            values,
            dtype: DType::F64,
        }
    }

    /// Unwrap phase angles by changing deltas > discont to their 2*pi complement (np.unwrap).
    pub fn unwrap(&self, discont: Option<f64>) -> Result<Self, UFuncError> {
        if self.shape.len() != 1 {
            return Err(UFuncError::Msg(
                "unwrap: only 1-D arrays supported".to_string(),
            ));
        }
        let disc = discont.unwrap_or(std::f64::consts::PI);
        let mut values = self.values.clone();
        let mut offset = 0.0;
        for i in 1..values.len() {
            let diff = values[i] - values[i - 1] + offset;
            offset = 0.0;
            if diff > disc {
                offset = -2.0
                    * std::f64::consts::PI
                    * ((diff + std::f64::consts::PI) / (2.0 * std::f64::consts::PI)).floor();
            } else if diff < -disc {
                offset = 2.0
                    * std::f64::consts::PI
                    * ((-diff + std::f64::consts::PI) / (2.0 * std::f64::consts::PI)).floor();
            }
            values[i] += offset;
        }
        Ok(Self {
            shape: self.shape.clone(),
            values,
            dtype: DType::F64,
        })
    }

    // ── index utilities ────────

    /// Convert a flat index into a tuple of coordinate arrays (np.unravel_index).
    pub fn unravel_index(indices: &Self, shape: &[usize]) -> Result<Vec<Self>, UFuncError> {
        let strides = c_strides_elems(shape);
        let ndim = shape.len();
        let total: usize = shape.iter().product();
        let mut result: Vec<Vec<f64>> = vec![Vec::with_capacity(indices.values.len()); ndim];
        for &idx_f in &indices.values {
            let idx = idx_f as usize;
            if idx >= total {
                return Err(UFuncError::Msg(format!(
                    "unravel_index: index {idx} out of bounds for shape {shape:?}"
                )));
            }
            let mut rem = idx;
            for d in 0..ndim {
                result[d].push((rem / strides[d]) as f64);
                rem %= strides[d];
            }
        }
        let n = indices.values.len();
        Ok(result
            .into_iter()
            .map(|vals| Self {
                shape: vec![n],
                values: vals,
                dtype: DType::I64,
            })
            .collect())
    }

    /// Convert coordinate arrays into a flat index (np.ravel_multi_index).
    pub fn ravel_multi_index(multi_index: &[&Self], shape: &[usize]) -> Result<Self, UFuncError> {
        let ndim = shape.len();
        if multi_index.len() != ndim {
            return Err(UFuncError::Msg(format!(
                "ravel_multi_index: {} index arrays for {} dimensions",
                multi_index.len(),
                ndim
            )));
        }
        let n = multi_index[0].values.len();
        for arr in multi_index {
            if arr.values.len() != n {
                return Err(UFuncError::Msg(
                    "ravel_multi_index: all index arrays must have same length".to_string(),
                ));
            }
        }
        let strides = c_strides_elems(shape);
        let mut values = Vec::with_capacity(n);
        for i in 0..n {
            let mut flat = 0usize;
            for d in 0..ndim {
                let idx = multi_index[d].values[i] as usize;
                if idx >= shape[d] {
                    return Err(UFuncError::Msg(format!(
                        "ravel_multi_index: index {idx} out of bounds for axis {d} with size {}",
                        shape[d]
                    )));
                }
                flat += idx * strides[d];
            }
            values.push(flat as f64);
        }
        Ok(Self {
            shape: vec![n],
            values,
            dtype: DType::I64,
        })
    }

    /// Open meshgrid: return sparse arrays for N-D indexing (np.ogrid equivalent).
    /// Each returned array has shape with 1s everywhere except along its own axis.
    pub fn ogrid(slices: &[(f64, f64, usize)]) -> Vec<Self> {
        let ndim = slices.len();
        slices
            .iter()
            .enumerate()
            .map(|(ax, &(start, stop, num))| {
                let step = if num <= 1 {
                    0.0
                } else {
                    (stop - start) / (num as f64 - 1.0)
                };
                let vals: Vec<f64> = (0..num).map(|i| start + i as f64 * step).collect();
                let mut shape = vec![1usize; ndim];
                shape[ax] = num;
                Self {
                    shape,
                    values: vals,
                    dtype: DType::F64,
                }
            })
            .collect()
    }

    /// Dense meshgrid: return full arrays for N-D indexing (np.mgrid equivalent).
    /// Each returned array has the full output shape.
    pub fn mgrid(slices: &[(f64, f64, usize)]) -> Vec<Self> {
        let out_shape: Vec<usize> = slices.iter().map(|s| s.2).collect();
        let out_count: usize = out_shape.iter().product();
        let out_strides = c_strides_elems(&out_shape);
        slices
            .iter()
            .enumerate()
            .map(|(ax, &(start, stop, num))| {
                let step = if num <= 1 {
                    0.0
                } else {
                    (stop - start) / (num as f64 - 1.0)
                };
                let values: Vec<f64> = (0..out_count)
                    .map(|flat| {
                        let coord = (flat / out_strides[ax]) % out_shape[ax];
                        start + coord as f64 * step
                    })
                    .collect();
                Self {
                    shape: out_shape.clone(),
                    values,
                    dtype: DType::F64,
                }
            })
            .collect()
    }

    /// Broadcast an array to a new shape (np.broadcast_to).
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Result<Self, UFuncError> {
        let ndim = target_shape.len();
        if self.shape.len() > ndim {
            return Err(UFuncError::Msg(format!(
                "broadcast_to: cannot broadcast shape {:?} to {:?}",
                self.shape, target_shape
            )));
        }
        // Left-pad shape with 1s
        let mut padded = vec![1usize; ndim - self.shape.len()];
        padded.extend_from_slice(&self.shape);
        // Validate
        for (d, (&s, &t)) in padded.iter().zip(target_shape).enumerate() {
            if s != t && s != 1 {
                return Err(UFuncError::Msg(format!(
                    "broadcast_to: operands could not be broadcast, shape {:?} vs {:?} at axis {d}",
                    self.shape, target_shape
                )));
            }
        }
        let src_strides = c_strides_elems(&padded);
        let out_strides = c_strides_elems(target_shape);
        let out_count: usize = target_shape.iter().product();
        let values: Vec<f64> = (0..out_count)
            .map(|flat| {
                let mut remainder = flat;
                let mut src_flat = 0;
                for d in 0..ndim {
                    let coord = remainder / out_strides[d];
                    remainder %= out_strides[d];
                    if padded[d] > 1 {
                        src_flat += coord * src_strides[d];
                    }
                }
                self.values[src_flat]
            })
            .collect();
        Ok(Self {
            shape: target_shape.to_vec(),
            values,
            dtype: self.dtype,
        })
    }

    /// Enumerate all elements with their N-d index (np.ndenumerate).
    /// Returns (index_vec, value) pairs in C-order.
    pub fn ndenumerate(&self) -> Vec<(Vec<usize>, f64)> {
        let strides = c_strides_elems(&self.shape);
        self.values
            .iter()
            .enumerate()
            .map(|(flat, &val)| {
                let mut idx = Vec::with_capacity(self.shape.len());
                let mut rem = flat;
                for &s in &strides {
                    idx.push(rem / s);
                    rem %= s;
                }
                (idx, val)
            })
            .collect()
    }

    /// All N-d indices for a given shape (np.ndindex).
    pub fn ndindex(shape: &[usize]) -> Vec<Vec<usize>> {
        let total: usize = shape.iter().product();
        let strides = c_strides_elems(shape);
        (0..total)
            .map(|flat| {
                let mut idx = Vec::with_capacity(shape.len());
                let mut rem = flat;
                for &s in &strides {
                    idx.push(rem / s);
                    rem %= s;
                }
                idx
            })
            .collect()
    }

    /// Flat iterator over elements in C-order (ndarray.flat).
    pub fn flat(&self) -> &[f64] {
        &self.values
    }

    /// Count the number of non-zero elements (np.count_nonzero).
    pub fn count_nonzero(&self) -> usize {
        self.values.iter().filter(|&&v| v != 0.0).count()
    }

    /// Pack the elements of a binary-valued array into bits (np.packbits).
    /// Input values > 0 are treated as 1. Returns a 1-D u8 array
    /// stored as f64 values (each value is a packed byte, 0–255).
    pub fn packbits(&self) -> Self {
        let n = self.values.len();
        let nbytes = n.div_ceil(8);
        let mut packed = vec![0.0; nbytes];
        for (i, &v) in self.values.iter().enumerate() {
            if v > 0.0 {
                let byte_idx = i / 8;
                let bit_idx = 7 - (i % 8); // MSB first, matching NumPy default
                packed[byte_idx] += (1u8 << bit_idx) as f64;
            }
        }
        Self {
            shape: vec![nbytes],
            values: packed,
            dtype: DType::U8,
        }
    }

    /// Unpack elements of a uint8 array into a binary array (np.unpackbits).
    /// Each input value (0–255) is expanded to 8 bits. Returns a 1-D array of 0.0/1.0.
    pub fn unpackbits(&self) -> Self {
        let mut bits = Vec::with_capacity(self.values.len() * 8);
        for &v in &self.values {
            let byte = v as u8;
            for bit in (0..8).rev() {
                bits.push(if byte & (1 << bit) != 0 { 1.0 } else { 0.0 });
            }
        }
        let n = bits.len();
        Self {
            shape: vec![n],
            values: bits,
            dtype: DType::U8,
        }
    }
}

/// Options controlling array string representation (np.set_printoptions).
#[derive(Debug, Clone)]
pub struct PrintOptions {
    /// Maximum number of elements to show before summarizing with "...".
    pub threshold: usize,
    /// Number of elements at the beginning and end when summarizing.
    pub edgeitems: usize,
    /// Number of digits of precision for floating-point output.
    pub precision: usize,
    /// Whether to suppress small floating-point values to zero.
    pub suppress: bool,
    /// Separator between elements.
    pub separator: String,
}

impl Default for PrintOptions {
    fn default() -> Self {
        Self {
            threshold: 1000,
            edgeitems: 3,
            precision: 8,
            suppress: false,
            separator: " ".to_string(),
        }
    }
}

impl UFuncArray {
    /// Format array as a string (np.array2string).
    pub fn array2string(&self, opts: &PrintOptions) -> String {
        let total: usize = self.shape.iter().product();
        if self.shape.is_empty() {
            // Scalar
            return format_value(self.values.first().copied().unwrap_or(0.0), opts);
        }
        if self.shape.len() == 1 {
            return format_1d(&self.values, opts, total);
        }
        format_nd(self, opts, 0, 0)
    }

    /// Return a string representation like NumPy's repr (np.array_repr).
    pub fn array_repr(&self, opts: &PrintOptions) -> String {
        let content = self.array2string(opts);
        let dtype_str = match self.dtype {
            DType::F64 => "",
            DType::Bool => "bool",
            DType::I8 => "int8",
            DType::I16 => "int16",
            DType::I32 => "int32",
            DType::I64 => "int64",
            DType::U8 => "uint8",
            DType::U16 => "uint16",
            DType::U32 => "uint32",
            DType::U64 => "uint64",
            DType::F32 => "float32",
        };
        if dtype_str.is_empty() {
            format!("array({content})")
        } else {
            format!("array({content}, dtype={dtype_str})")
        }
    }
}

fn format_value(v: f64, opts: &PrintOptions) -> String {
    if opts.suppress && v.abs() < 1e-15 {
        "0.".to_string()
    } else if v == v.floor() && v.abs() < 1e15 && !v.is_nan() && !v.is_infinite() {
        format!("{v:.0}.")
    } else {
        format!("{v:.prec$}", prec = opts.precision)
    }
}

fn format_1d(values: &[f64], opts: &PrintOptions, total: usize) -> String {
    let mut parts = Vec::new();
    if total <= opts.threshold {
        for v in values {
            parts.push(format_value(*v, opts));
        }
    } else {
        for v in &values[..opts.edgeitems] {
            parts.push(format_value(*v, opts));
        }
        parts.push("...".to_string());
        for v in &values[values.len() - opts.edgeitems..] {
            parts.push(format_value(*v, opts));
        }
    }
    format!("[{}]", parts.join(&format!(",{}", opts.separator)))
}

fn format_nd(arr: &UFuncArray, opts: &PrintOptions, axis: usize, offset: usize) -> String {
    if axis == arr.shape.len() - 1 {
        let n = arr.shape[axis];
        let values = &arr.values[offset..offset + n];
        return format_1d(values, opts, n);
    }
    let inner_size: usize = arr.shape[axis + 1..].iter().product();
    let n = arr.shape[axis];
    let mut parts = Vec::new();
    if n <= opts.threshold {
        for i in 0..n {
            parts.push(format_nd(arr, opts, axis + 1, offset + i * inner_size));
        }
    } else {
        for i in 0..opts.edgeitems {
            parts.push(format_nd(arr, opts, axis + 1, offset + i * inner_size));
        }
        parts.push("...".to_string());
        for i in (n - opts.edgeitems)..n {
            parts.push(format_nd(arr, opts, axis + 1, offset + i * inner_size));
        }
    }
    let sep = format!(",\n{}", " ".repeat(axis + 1));
    format!("[{}]", parts.join(&sep))
}

/// Linear interpolation for percentile on a sorted slice (NumPy default method).
fn interpolate_percentile(sorted: &[f64], fraction: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = fraction * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = lo + 1;
    if hi >= n {
        sorted[n - 1]
    } else {
        let t = idx - lo as f64;
        sorted[lo] * (1.0 - t) + sorted[hi] * t
    }
}

/// Compute C-order strides in elements (not bytes) for a given shape.
fn c_strides_elems(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[derive(Debug, Clone, PartialEq)]
pub enum UFuncError {
    Shape(ShapeError),
    InvalidInputLength { expected: usize, actual: usize },
    AxisOutOfBounds { axis: isize, ndim: usize },
    EmptyReduction { op: &'static str },
    SignatureConflict { sig: String, signature: String },
    SignatureParse { detail: String },
    FixedSignatureInvalid { detail: String },
    OverridePrecedenceViolation { detail: String },
    LoopRegistryInvalid { detail: String },
    Msg(String),
}

impl std::fmt::Display for UFuncError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape(err) => write!(f, "shape error: {err}"),
            Self::InvalidInputLength { expected, actual } => {
                write!(
                    f,
                    "invalid input length expected={expected} actual={actual}"
                )
            }
            Self::AxisOutOfBounds { axis, ndim } => {
                write!(f, "axis {axis} out of bounds for ndim={ndim}")
            }
            Self::EmptyReduction { op } => {
                write!(f, "attempt to get {op} of an empty sequence")
            }
            Self::SignatureConflict { sig, signature } => {
                write!(
                    f,
                    "ufunc signature conflict: sig='{sig}' differs from signature='{signature}'"
                )
            }
            Self::SignatureParse { detail } => {
                write!(f, "ufunc signature parse failed: {detail}")
            }
            Self::FixedSignatureInvalid { detail } => {
                write!(f, "ufunc fixed signature invalid: {detail}")
            }
            Self::OverridePrecedenceViolation { detail } => {
                write!(f, "ufunc override precedence violation: {detail}")
            }
            Self::LoopRegistryInvalid { detail } => {
                write!(f, "ufunc loop registry invalid: {detail}")
            }
            Self::Msg(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for UFuncError {}

impl UFuncError {
    #[must_use]
    pub fn reason_code(&self) -> &'static str {
        match self {
            Self::Shape(_) => "ufunc_shape_contract_violation",
            Self::InvalidInputLength { .. } => "ufunc_invalid_input_length",
            Self::AxisOutOfBounds { .. } => "ufunc_axis_out_of_bounds",
            Self::EmptyReduction { .. } => "ufunc_reduce_axis_contract",
            Self::SignatureConflict { .. } => "ufunc_signature_conflict",
            Self::SignatureParse { .. } => "ufunc_signature_parse_failed",
            Self::FixedSignatureInvalid { .. } => "ufunc_fixed_signature_invalid",
            Self::OverridePrecedenceViolation { .. } => "ufunc_override_precedence_violation",
            Self::LoopRegistryInvalid { .. } => "ufunc_loop_registry_invalid",
            Self::Msg(_) => "ufunc_operation_error",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UFuncLogRecord {
    pub fixture_id: String,
    pub seed: u64,
    pub mode: UFuncRuntimeMode,
    pub env_fingerprint: String,
    pub artifact_refs: Vec<String>,
    pub reason_code: String,
    pub passed: bool,
}

impl UFuncLogRecord {
    #[must_use]
    pub fn is_replay_complete(&self) -> bool {
        !self.fixture_id.trim().is_empty()
            && !self.mode.as_str().is_empty()
            && !self.env_fingerprint.trim().is_empty()
            && !self.reason_code.trim().is_empty()
            && !self.artifact_refs.is_empty()
            && self
                .artifact_refs
                .iter()
                .all(|artifact| !artifact.trim().is_empty())
    }
}

#[must_use]
fn canonicalize_signature_groups(groups: &[Vec<String>]) -> String {
    groups
        .iter()
        .map(|group| format!("({})", group.join(",")))
        .collect::<Vec<_>>()
        .join(",")
}

fn validate_core_dim_identifier(token: &str) -> Result<(), UFuncError> {
    let mut chars = token.chars();
    let Some(first) = chars.next() else {
        return Err(UFuncError::SignatureParse {
            detail: "signature dimension token must not be empty".to_string(),
        });
    };

    if !(first.is_ascii_alphabetic() || first == '_') {
        return Err(UFuncError::SignatureParse {
            detail: format!("signature dimension '{}' must start with [A-Za-z_]", token),
        });
    }

    if chars.any(|ch| !(ch.is_ascii_alphanumeric() || ch == '_')) {
        return Err(UFuncError::SignatureParse {
            detail: format!(
                "signature dimension '{}' must contain only [A-Za-z0-9_]",
                token
            ),
        });
    }

    Ok(())
}

fn parse_signature_dims(raw: &str) -> Result<Vec<String>, UFuncError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }

    let mut dims = Vec::new();
    for token in trimmed.split(',') {
        let dim = token.trim();
        if dim.is_empty() {
            return Err(UFuncError::SignatureParse {
                detail: "signature dimension list contains an empty token".to_string(),
            });
        }
        validate_core_dim_identifier(dim)?;
        dims.push(dim.to_string());
    }
    Ok(dims)
}

fn parse_signature_groups(raw: &str, side: &str) -> Result<Vec<Vec<String>>, UFuncError> {
    let bytes = raw.as_bytes();
    let mut idx = 0usize;
    let mut groups = Vec::new();

    while idx < bytes.len() {
        while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if idx >= bytes.len() {
            break;
        }

        if bytes[idx] != b'(' {
            return Err(UFuncError::SignatureParse {
                detail: format!("{side} must start tuple groups with '(' at byte {idx}"),
            });
        }
        idx += 1;
        let tuple_start = idx;
        let mut depth = 1usize;
        while idx < bytes.len() && depth > 0 {
            match bytes[idx] {
                b'(' => depth += 1,
                b')' => depth -= 1,
                _ => {}
            }
            idx += 1;
        }

        if depth != 0 {
            return Err(UFuncError::SignatureParse {
                detail: format!("{side} has an unclosed tuple group"),
            });
        }

        let tuple_body = &raw[tuple_start..idx - 1];
        if tuple_body.contains('(') || tuple_body.contains(')') {
            return Err(UFuncError::SignatureParse {
                detail: format!("{side} contains nested tuple groups, which are unsupported"),
            });
        }
        groups.push(parse_signature_dims(tuple_body)?);

        while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if idx >= bytes.len() {
            break;
        }

        if bytes[idx] != b',' {
            return Err(UFuncError::SignatureParse {
                detail: format!("{side} expected ',' between tuple groups at byte {idx}"),
            });
        }
        idx += 1;
    }

    if groups.is_empty() {
        return Err(UFuncError::SignatureParse {
            detail: format!("{side} must contain at least one tuple group"),
        });
    }
    Ok(groups)
}

#[must_use]
fn contiguous_strides_elems(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut strides = vec![0usize; shape.len()];
    let mut stride = 1usize;
    for (idx, &dim) in shape.iter().enumerate().rev() {
        strides[idx] = stride;
        stride = stride.saturating_mul(dim);
    }
    strides
}

#[must_use]
fn aligned_broadcast_axis_steps(
    out_ndim: usize,
    src_shape: &[usize],
    src_strides: &[usize],
) -> Vec<usize> {
    if out_ndim == 0 {
        return Vec::new();
    }

    let mut axis_steps = vec![0usize; out_ndim];
    let offset = out_ndim - src_shape.len();

    for (axis, (&dim, &stride)) in src_shape.iter().zip(src_strides).enumerate() {
        axis_steps[axis + offset] = if dim == 1 { 0 } else { stride };
    }

    axis_steps
}

fn reduce_sum_axis_contiguous(
    values: &[f64],
    shape: &[usize],
    axis: usize,
    out_values: &mut [f64],
) {
    debug_assert!(axis < shape.len());
    if out_values.is_empty() {
        return;
    }

    let axis_len = shape[axis];
    if axis_len == 0 {
        return;
    }

    let inner = shape[axis + 1..].iter().copied().product::<usize>();
    let outer = shape[..axis].iter().copied().product::<usize>();

    let mut out_flat = 0usize;
    for outer_idx in 0..outer {
        let base = outer_idx * axis_len * inner;
        for inner_idx in 0..inner {
            let mut sum = 0.0f64;
            let mut offset = base + inner_idx;
            for _ in 0..axis_len {
                sum += values[offset];
                offset += inner;
            }
            out_values[out_flat] = sum;
            out_flat += 1;
        }
    }
}

fn reduce_fold_axis_contiguous(
    values: &[f64],
    shape: &[usize],
    axis: usize,
    out_values: &mut [f64],
    fold: impl Fn(f64, f64) -> f64,
) {
    debug_assert!(axis < shape.len());
    if out_values.is_empty() {
        return;
    }

    let axis_len = shape[axis];
    if axis_len == 0 {
        return;
    }

    let inner = shape[axis + 1..].iter().copied().product::<usize>();
    let outer = shape[..axis].iter().copied().product::<usize>();

    let mut out_flat = 0usize;
    for outer_idx in 0..outer {
        let base = outer_idx * axis_len * inner;
        for inner_idx in 0..inner {
            let mut offset = base + inner_idx;
            let mut acc = values[offset];
            offset += inner;
            for _ in 1..axis_len {
                acc = fold(acc, values[offset]);
                offset += inner;
            }
            out_values[out_flat] = acc;
            out_flat += 1;
        }
    }
}

fn cumulate_axis(
    values: &[f64],
    shape: &[usize],
    axis: usize,
    identity: f64,
    fold: impl Fn(f64, f64) -> f64,
) -> Result<Vec<f64>, UFuncError> {
    debug_assert!(axis < shape.len());
    let total = element_count(shape).map_err(UFuncError::Shape)?;
    let mut out = vec![0.0; total];

    let axis_len = shape[axis];
    if axis_len == 0 || total == 0 {
        return Ok(out);
    }

    let inner: usize = shape[axis + 1..].iter().copied().product();
    let outer: usize = shape[..axis].iter().copied().product();

    for outer_idx in 0..outer {
        let base = outer_idx * axis_len * inner;
        for inner_idx in 0..inner {
            let mut acc = identity;
            let mut offset = base + inner_idx;
            for _ in 0..axis_len {
                acc = fold(acc, values[offset]);
                out[offset] = acc;
                offset += inner;
            }
        }
    }

    Ok(out)
}

fn reduce_var_axis_contiguous(
    values: &[f64],
    shape: &[usize],
    axis: usize,
    means: &[f64],
    out_values: &mut [f64],
) {
    debug_assert!(axis < shape.len());
    if out_values.is_empty() {
        return;
    }

    let axis_len = shape[axis];
    if axis_len == 0 {
        return;
    }

    let inner = shape[axis + 1..].iter().copied().product::<usize>();
    let outer = shape[..axis].iter().copied().product::<usize>();

    let mut out_flat = 0usize;
    for outer_idx in 0..outer {
        let base = outer_idx * axis_len * inner;
        for inner_idx in 0..inner {
            let mean = means[out_flat];
            let mut sum_sq = 0.0f64;
            let mut offset = base + inner_idx;
            for _ in 0..axis_len {
                let diff = values[offset] - mean;
                sum_sq += diff * diff;
                offset += inner;
            }
            out_values[out_flat] = sum_sq;
            out_flat += 1;
        }
    }
}

fn reduce_argfold_axis_contiguous(
    values: &[f64],
    shape: &[usize],
    axis: usize,
    out_values: &mut [f64],
    is_better: impl Fn(f64, f64) -> bool,
) {
    debug_assert!(axis < shape.len());
    if out_values.is_empty() {
        return;
    }

    let axis_len = shape[axis];
    if axis_len == 0 {
        return;
    }

    let inner = shape[axis + 1..].iter().copied().product::<usize>();
    let outer = shape[..axis].iter().copied().product::<usize>();

    let mut out_flat = 0usize;
    for outer_idx in 0..outer {
        let base = outer_idx * axis_len * inner;
        for inner_idx in 0..inner {
            let mut offset = base + inner_idx;
            let mut best_val = values[offset];
            let mut best_idx = 0usize;
            offset += inner;
            for k in 1..axis_len {
                let cur = values[offset];
                if cur.is_nan() {
                    if !best_val.is_nan() {
                        best_val = cur;
                        best_idx = k;
                    }
                } else if !best_val.is_nan() && is_better(cur, best_val) {
                    best_val = cur;
                    best_idx = k;
                }
                offset += inner;
            }
            out_values[out_flat] = best_idx as f64;
            out_flat += 1;
        }
    }
}

/// Modified Bessel function of the first kind, order 0.
/// Uses the polynomial approximation from Abramowitz and Stegun.
fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let t = (ax / 3.75).powi(2);
        1.0 + t
            * (3.5156229
                + t * (3.0899424
                    + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))
    } else {
        let t = 3.75 / ax;
        let e = ax.exp() / ax.sqrt();
        e * (0.39894228
            + t * (0.01328592
                + t * (0.00225319
                    + t * (-0.00157565
                        + t * (0.00916281
                            + t * (-0.02057706
                                + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377))))))))
    }
}

/// Mixed-radix Bluestein/Chirp-Z FFT supporting arbitrary lengths.
/// For power-of-two lengths, uses Cooley-Tukey DIT.
/// For non-power-of-two, uses Bluestein's algorithm (zero-pad to power-of-two).
fn fft_dit(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    if n <= 1 {
        if inverse && n == 1 {
            // No scaling needed for single element
        }
        return;
    }
    // Check if n is a power of two
    if n & (n - 1) == 0 {
        fft_pow2(re, im, inverse);
    } else {
        // Bluestein: convert length-n DFT to circular convolution of length m (power of 2)
        let m = (2 * n - 1).next_power_of_two();
        let sign = if inverse { 1.0 } else { -1.0 };

        // Chirp sequence: w[k] = exp(sign * i * pi * k^2 / n)
        let mut chirp_re = vec![0.0; n];
        let mut chirp_im = vec![0.0; n];
        for k in 0..n {
            let angle = sign * std::f64::consts::PI * (k * k) as f64 / n as f64;
            chirp_re[k] = angle.cos();
            chirp_im[k] = angle.sin();
        }

        // a[k] = x[k] * conj(chirp[k])  — zero-padded to m
        let mut a_re = vec![0.0; m];
        let mut a_im = vec![0.0; m];
        for k in 0..n {
            a_re[k] = re[k] * chirp_re[k] + im[k] * chirp_im[k];
            a_im[k] = im[k] * chirp_re[k] - re[k] * chirp_im[k];
        }

        // b[k] = chirp[k] for k=0..n-1, chirp[m-k] = chirp[k] for k=1..n-1, rest zero
        let mut b_re = vec![0.0; m];
        let mut b_im = vec![0.0; m];
        b_re[0] = chirp_re[0];
        b_im[0] = chirp_im[0];
        for k in 1..n {
            b_re[k] = chirp_re[k];
            b_im[k] = chirp_im[k];
            b_re[m - k] = chirp_re[k];
            b_im[m - k] = chirp_im[k];
        }

        // FFT both, multiply, IFFT
        fft_pow2(&mut a_re, &mut a_im, false);
        fft_pow2(&mut b_re, &mut b_im, false);

        // Pointwise complex multiply
        for k in 0..m {
            let tr = a_re[k] * b_re[k] - a_im[k] * b_im[k];
            let ti = a_re[k] * b_im[k] + a_im[k] * b_re[k];
            a_re[k] = tr;
            a_im[k] = ti;
        }

        fft_pow2(&mut a_re, &mut a_im, true);

        // Extract result: X[k] = conj(chirp[k]) * conv[k]
        for k in 0..n {
            let cr = chirp_re[k];
            let ci = chirp_im[k];
            // conj(chirp) = (cr, -ci)
            re[k] = a_re[k] * cr + a_im[k] * ci;
            im[k] = a_im[k] * cr - a_re[k] * ci;
        }

        if inverse {
            let scale = 1.0 / n as f64;
            for k in 0..n {
                re[k] *= scale;
                im[k] *= scale;
            }
        }
    }
}

/// Cooley-Tukey radix-2 DIT FFT. Length MUST be a power of two.
fn fft_pow2(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation
    let mut j: usize = 0;
    for i in 0..n {
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
        let mut m = n >> 1;
        while m >= 1 && j >= m {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    // Butterfly passes
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle_step = sign * std::f64::consts::TAU / len as f64;
        let wn_re = angle_step.cos();
        let wn_im = angle_step.sin();
        let mut start = 0;
        while start < n {
            let mut w_re = 1.0;
            let mut w_im = 0.0;
            for k in 0..half {
                let even = start + k;
                let odd = start + k + half;
                let tr = w_re * re[odd] - w_im * im[odd];
                let ti = w_re * im[odd] + w_im * re[odd];
                re[odd] = re[even] - tr;
                im[odd] = im[even] - ti;
                re[even] += tr;
                im[even] += ti;
                let new_w_re = w_re * wn_re - w_im * wn_im;
                let new_w_im = w_re * wn_im + w_im * wn_re;
                w_re = new_w_re;
                w_im = new_w_im;
            }
            start += len;
        }
        len <<= 1;
    }

    if inverse {
        let scale = 1.0 / n as f64;
        for i in 0..n {
            re[i] *= scale;
            im[i] *= scale;
        }
    }
}

/// Reflect index for 'reflect' pad mode (no edge duplication).
/// Maps indices outside [0, n) using reflection at boundaries.
fn reflect_index(idx: isize, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let period = 2 * (n as isize - 1);
    if period == 0 {
        return 0;
    }
    let mut i = idx.rem_euclid(period);
    if i >= n as isize {
        i = period - i;
    }
    i as usize
}

/// Symmetric index for 'symmetric' pad mode (edge duplication).
fn symmetric_index(idx: isize, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let period = 2 * n as isize;
    if period == 0 {
        return 0;
    }
    let mut i = idx.rem_euclid(period);
    if i >= n as isize {
        i = period - 1 - i;
    }
    i as usize
}

#[must_use]
fn reduced_shape(shape: &[usize], axis: usize, keepdims: bool) -> Vec<usize> {
    if keepdims {
        shape
            .iter()
            .enumerate()
            .map(|(idx, &dim)| if idx == axis { 1 } else { dim })
            .collect()
    } else {
        shape
            .iter()
            .enumerate()
            .filter_map(|(idx, &dim)| (idx != axis).then_some(dim))
            .collect()
    }
}

fn normalize_axis(axis: isize, ndim: usize) -> Result<usize, UFuncError> {
    let ndim_i128 = i128::try_from(ndim).expect("ndim should always fit into i128");
    let axis_i128 = i128::try_from(axis).expect("isize should always fit into i128");
    let normalized = if axis_i128 < 0 {
        ndim_i128 + axis_i128
    } else {
        axis_i128
    };
    if normalized < 0 || normalized >= ndim_i128 {
        return Err(UFuncError::AxisOutOfBounds { axis, ndim });
    }
    usize::try_from(normalized).map_err(|_| UFuncError::AxisOutOfBounds { axis, ndim })
}

#[cfg(test)]
mod tests {
    use super::{
        BinaryOp, PrintOptions, UFUNC_PACKET_REASON_CODES, UFuncArray, UFuncError, UFuncLogRecord,
        UFuncRuntimeMode, UnaryOp, normalize_signature_keywords, parse_gufunc_signature,
        plan_binary_dispatch, register_custom_loop, validate_override_payload_class,
    };
    use fnp_dtype::{DType, promote};
    use fnp_ndarray::broadcast_shape;

    fn packet005_artifacts() -> Vec<String> {
        vec![
            "artifacts/phase2c/FNP-P2C-005/fixture_manifest.json".to_string(),
            "artifacts/phase2c/FNP-P2C-005/parity_gate.yaml".to_string(),
        ]
    }

    #[test]
    fn broadcasted_add_matches_expected_values() {
        let lhs = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("lhs");
        let rhs = UFuncArray::new(vec![3], vec![10.0, 20.0, 30.0], DType::F64).expect("rhs");

        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Add)
            .expect("broadcasted add");

        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.values(), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn scalar_broadcast_mul() {
        let lhs = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::scalar(0.5, DType::F64);

        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Mul)
            .expect("scalar mul");

        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.values(), &[0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn division_by_zero_yields_infinity_semantics() {
        let lhs = UFuncArray::new(vec![2], vec![1.0, -1.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![1], vec![0.0], DType::F64).expect("rhs");

        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Div)
            .expect("div should execute");

        assert!(out.values()[0].is_infinite() && out.values()[0].is_sign_positive());
        assert!(out.values()[1].is_infinite() && out.values()[1].is_sign_negative());
    }

    #[test]
    fn reduce_sum_axis_none_keepdims_false() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");

        let out = arr.reduce_sum(None, false).expect("sum");
        assert_eq!(out.shape(), &[]);
        assert_eq!(out.values(), &[21.0]);
    }

    #[test]
    fn reduce_sum_axis_none_keepdims_true() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");

        let out = arr.reduce_sum(None, true).expect("sum");
        assert_eq!(out.shape(), &[1, 1]);
        assert_eq!(out.values(), &[21.0]);
    }

    #[test]
    fn reduce_sum_axis_keepdims_variants() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");

        let out_drop = arr.reduce_sum(Some(1), false).expect("sum axis=1");
        assert_eq!(out_drop.shape(), &[2]);
        assert_eq!(out_drop.values(), &[6.0, 15.0]);

        let out_keep = arr.reduce_sum(Some(1), true).expect("sum axis=1 keepdims");
        assert_eq!(out_keep.shape(), &[2, 1]);
        assert_eq!(out_keep.values(), &[6.0, 15.0]);
    }

    #[test]
    fn reduce_sum_axis_zero_preserves_c_order() {
        let arr = UFuncArray::new(
            vec![2, 3, 2],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            DType::F64,
        )
        .expect("arr");

        let out_drop = arr.reduce_sum(Some(0), false).expect("sum axis=0");
        assert_eq!(out_drop.shape(), &[3, 2]);
        assert_eq!(out_drop.values(), &[8.0, 10.0, 12.0, 14.0, 16.0, 18.0]);

        let out_keep = arr.reduce_sum(Some(0), true).expect("sum axis=0 keepdims");
        assert_eq!(out_keep.shape(), &[1, 3, 2]);
        assert_eq!(out_keep.values(), &[8.0, 10.0, 12.0, 14.0, 16.0, 18.0]);
    }

    #[test]
    fn reduce_sum_empty_axis_returns_zero_initialized_outputs() {
        let arr = UFuncArray::new(vec![2, 0, 3], Vec::new(), DType::F64).expect("arr");
        let out = arr.reduce_sum(Some(1), false).expect("sum axis=1");
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.values(), &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn mixed_dtype_high_rank_add_promotes_and_broadcasts() {
        let lhs =
            UFuncArray::new(vec![2, 1, 2], vec![1.0, 2.0, 3.0, 4.0], DType::I32).expect("lhs");
        let rhs = UFuncArray::new(vec![1, 3, 1], vec![0.5, -1.0, 2.0], DType::F32).expect("rhs");

        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Add)
            .expect("broadcasted mixed-dtype add");

        assert_eq!(out.shape(), &[2, 3, 2]);
        assert_eq!(out.dtype(), DType::F64);
        assert_eq!(
            out.values(),
            &[1.5, 2.5, 0.0, 1.0, 3.0, 4.0, 3.5, 4.5, 2.0, 3.0, 5.0, 6.0]
        );
    }

    #[test]
    fn mixed_dtype_bool_i64_mul_promotes_to_i64() {
        let lhs =
            UFuncArray::new(vec![2, 1, 2], vec![1.0, 0.0, 0.0, 1.0], DType::Bool).expect("lhs");
        let rhs = UFuncArray::new(vec![1, 3, 1], vec![10.0, -2.0, 7.0], DType::I64).expect("rhs");

        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Mul)
            .expect("broadcasted mixed-dtype mul");

        assert_eq!(out.shape(), &[2, 3, 2]);
        assert_eq!(out.dtype(), DType::I64);
        assert_eq!(
            out.values(),
            &[
                10.0, 0.0, -2.0, 0.0, 7.0, 0.0, 0.0, 10.0, 0.0, -2.0, 0.0, 7.0
            ]
        );
    }

    #[test]
    fn reduce_sum_axis_one_keepdims_promotes_i32_to_i64() {
        let arr = UFuncArray::new(
            vec![2, 2, 2],
            vec![-3.0, 1.0, 5.0, -7.0, 9.0, 11.0, -13.0, 15.0],
            DType::I32,
        )
        .expect("arr");

        let out = arr.reduce_sum(Some(1), true).expect("sum axis=1 keepdims");
        assert_eq!(out.shape(), &[2, 1, 2]);
        assert_eq!(out.values(), &[2.0, -6.0, -4.0, 26.0]);
        assert_eq!(out.dtype(), DType::I64);
    }

    #[test]
    fn reduce_sum_promotes_bool_to_i64() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0], DType::Bool)
            .expect("arr");

        let out = arr.reduce_sum(None, false).expect("sum all");
        assert_eq!(out.values(), &[4.0]);
        assert_eq!(out.dtype(), DType::I64);

        let axis0 = arr.reduce_sum(Some(0), false).expect("sum axis=0");
        assert_eq!(axis0.values(), &[1.0, 1.0, 2.0]);
        assert_eq!(axis0.dtype(), DType::I64);
    }

    #[test]
    fn reduce_sum_preserves_f32_dtype() {
        let arr = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F32).expect("arr");

        let out = arr.reduce_sum(None, false).expect("sum all");
        assert_eq!(out.dtype(), DType::F32);
    }

    #[test]
    fn reduce_sum_preserves_i64_dtype() {
        let arr = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::I64).expect("arr");

        let out = arr.reduce_sum(None, false).expect("sum all");
        assert_eq!(out.dtype(), DType::I64);
    }

    #[test]
    fn reduce_sum_propagates_non_finite_values() {
        let arr = UFuncArray::new(
            vec![2, 2, 2],
            vec![
                1.0,
                f64::INFINITY,
                -1.0,
                2.0,
                3.0,
                f64::NEG_INFINITY,
                f64::NAN,
                4.0,
            ],
            DType::F64,
        )
        .expect("arr");

        let axis_one = arr.reduce_sum(Some(1), true).expect("sum axis=1 keepdims");
        assert_eq!(axis_one.shape(), &[2, 1, 2]);
        assert_eq!(axis_one.values()[0], 0.0);
        assert!(axis_one.values()[1].is_infinite() && axis_one.values()[1].is_sign_positive());
        assert!(axis_one.values()[2].is_nan());
        assert!(axis_one.values()[3].is_infinite() && axis_one.values()[3].is_sign_negative());

        let full = arr.reduce_sum(None, false).expect("sum all");
        assert_eq!(full.shape(), &[]);
        assert!(full.values()[0].is_nan());
    }

    #[test]
    fn invalid_input_length_is_rejected() {
        let err = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0], DType::F64)
            .expect_err("should reject length mismatch");
        assert!(matches!(err, UFuncError::InvalidInputLength { .. }));
    }

    #[test]
    fn normalize_signature_keywords_accepts_matching_sig_and_signature() {
        let normalized = normalize_signature_keywords(
            Some(" (batch,m),(m,n)->(batch,n) "),
            Some("(batch,m),(m,n)->(batch,n)"),
        )
        .expect("matching signature keywords should normalize");
        assert_eq!(normalized.as_deref(), Some("(batch,m),(m,n)->(batch,n)"));
    }

    #[test]
    fn parse_gufunc_signature_conflict_is_rejected() {
        let err = parse_gufunc_signature(Some("(i)->(i)"), Some("(j)->(j)"))
            .expect_err("conflicting sig/signature should fail");
        assert!(matches!(err, UFuncError::SignatureConflict { .. }));
        assert_eq!(err.reason_code(), "ufunc_signature_conflict");
    }

    #[test]
    fn parse_gufunc_signature_normalizes_grammar() {
        let signature = parse_gufunc_signature(Some(" (batch,m),(m,n) -> (batch,n) "), None)
            .expect("valid signature should parse")
            .expect("signature should be present");
        assert_eq!(signature.inputs.len(), 2);
        assert_eq!(signature.outputs.len(), 1);
        assert_eq!(signature.inputs[0], vec!["batch", "m"]);
        assert_eq!(signature.inputs[1], vec!["m", "n"]);
        assert_eq!(signature.outputs[0], vec!["batch", "n"]);
        assert_eq!(signature.canonical(), "(batch,m),(m,n)->(batch,n)");
    }

    #[test]
    fn parse_gufunc_signature_invalid_grammar_is_rejected() {
        let err = parse_gufunc_signature(Some("(i,j)->i,j"), None)
            .expect_err("invalid grammar must fail");
        assert!(matches!(err, UFuncError::SignatureParse { .. }));
        assert_eq!(err.reason_code(), "ufunc_signature_parse_failed");
    }

    #[test]
    fn dispatch_plan_is_deterministic() {
        let lhs = UFuncArray::new(
            vec![2, 1, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            DType::I32,
        )
        .expect("lhs");
        let rhs =
            UFuncArray::new(vec![1, 4, 1], vec![10.0, 20.0, 30.0, 40.0], DType::F32).expect("rhs");

        let first = plan_binary_dispatch(&lhs, &rhs).expect("dispatch plan");
        let second = plan_binary_dispatch(&lhs, &rhs).expect("dispatch plan repeat");
        assert_eq!(first, second);
        assert_eq!(first.out_shape, vec![2, 4, 3]);
        assert_eq!(first.out_count, 24);
        assert_eq!(first.out_dtype, DType::F64);
    }

    #[test]
    fn override_payload_validation_rejects_unknown_class() {
        let err = validate_override_payload_class("bogus_payload")
            .expect_err("unknown override payload class must fail");
        assert!(matches!(
            err,
            UFuncError::OverridePrecedenceViolation { .. }
        ));
        assert_eq!(err.reason_code(), "ufunc_override_precedence_violation");
    }

    #[test]
    fn custom_loop_registration_is_fail_closed() {
        let err = register_custom_loop("fused_add_loop")
            .expect_err("loop registration is unsupported in packet-D boundary");
        assert!(matches!(err, UFuncError::LoopRegistryInvalid { .. }));
        assert_eq!(err.reason_code(), "ufunc_loop_registry_invalid");
    }

    #[test]
    fn axis_out_of_bounds_is_rejected() {
        let arr = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).expect("arr");
        let err = arr
            .reduce_sum(Some(2), false)
            .expect_err("axis should be invalid");
        assert!(matches!(
            err,
            UFuncError::AxisOutOfBounds { axis: 2, ndim: 2 }
        ));
        assert_eq!(err.reason_code(), "ufunc_axis_out_of_bounds");
    }

    #[test]
    fn negative_axis_wraps_from_end() {
        let arr = UFuncArray::new(vec![2, 2, 2], (1..=8).map(f64::from).collect(), DType::F64)
            .expect("arr");
        let out = arr
            .reduce_sum(Some(-1), false)
            .expect("axis=-1 should be valid");
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.values(), &[3.0, 7.0, 11.0, 15.0]);
    }

    #[test]
    fn negative_axis_out_of_bounds_is_rejected() {
        let arr = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).expect("arr");
        let err = arr
            .reduce_sum(Some(-3), false)
            .expect_err("axis should be invalid");
        assert!(matches!(
            err,
            UFuncError::AxisOutOfBounds { axis: -3, ndim: 2 }
        ));
        assert_eq!(err.reason_code(), "ufunc_axis_out_of_bounds");
    }

    #[test]
    fn reason_code_registry_matches_packet_contract() {
        assert_eq!(
            UFUNC_PACKET_REASON_CODES,
            [
                "ufunc_shape_contract_violation",
                "ufunc_invalid_input_length",
                "ufunc_axis_out_of_bounds",
                "ufunc_division_by_zero_observed",
                "ufunc_broadcast_selector_determinism",
                "ufunc_reduce_keepdims_contract",
                "ufunc_reduce_axis_contract",
                "ufunc_scalar_broadcast_contract",
                "ufunc_dtype_promotion_contract",
                "ufunc_signature_conflict",
                "ufunc_signature_parse_failed",
                "ufunc_fixed_signature_invalid",
                "ufunc_override_precedence_violation",
                "gufunc_loop_exception_propagated",
                "ufunc_loop_registry_invalid",
                "ufunc_policy_unknown_metadata",
            ]
        );
    }

    #[test]
    fn ufunc_error_reason_codes_cover_adversarial_paths() {
        let len_err = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0], DType::F64)
            .expect_err("invalid length should fail");
        assert_eq!(len_err.reason_code(), "ufunc_invalid_input_length");

        let lhs = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("lhs");
        let rhs = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).expect("rhs");
        let shape_err = lhs
            .elementwise_binary(&rhs, BinaryOp::Add)
            .expect_err("incompatible broadcast should fail");
        assert_eq!(shape_err.reason_code(), "ufunc_shape_contract_violation");
    }

    #[test]
    fn elementwise_binary_property_grid_is_deterministic() {
        let cases = [
            (
                vec![2, 3],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                DType::F64,
                vec![3],
                vec![10.0, 20.0, 30.0],
                DType::F64,
            ),
            (
                vec![1, 3],
                vec![2.0, 4.0, 8.0],
                DType::I32,
                vec![2, 1, 3],
                vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
                DType::F32,
            ),
            (
                vec![],
                vec![4.0],
                DType::I64,
                vec![2, 2],
                vec![1.0, 2.0, 3.0, 4.0],
                DType::I32,
            ),
        ];

        let ops = [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div];

        for (lhs_shape, lhs_values, lhs_dtype, rhs_shape, rhs_values, rhs_dtype) in cases {
            let lhs =
                UFuncArray::new(lhs_shape.clone(), lhs_values.clone(), lhs_dtype).expect("lhs");
            let rhs =
                UFuncArray::new(rhs_shape.clone(), rhs_values.clone(), rhs_dtype).expect("rhs");
            let expected_dtype = promote(lhs_dtype, rhs_dtype);
            let expected_shape =
                broadcast_shape(&lhs_shape, &rhs_shape).expect("broadcast should work");

            for op in ops {
                let first = lhs
                    .elementwise_binary(&rhs, op)
                    .expect("operation should succeed");
                let second = lhs
                    .elementwise_binary(&rhs, op)
                    .expect("operation should be deterministic");
                assert_eq!(first.shape(), second.shape());
                assert_eq!(first.values(), second.values());
                assert_eq!(first.dtype(), second.dtype());
                assert_eq!(first.dtype(), expected_dtype);
                assert_eq!(first.shape(), expected_shape.as_slice());
            }
        }
    }

    #[test]
    fn reduce_sum_keepdims_shape_contract_holds() {
        let arr = UFuncArray::new(vec![2, 3, 4], (1..=24).map(f64::from).collect(), DType::F64)
            .expect("arr");

        for axis in 0_isize..3_isize {
            let keep = arr
                .reduce_sum(Some(axis), true)
                .expect("keepdims reduction should succeed");
            let drop = arr
                .reduce_sum(Some(axis), false)
                .expect("dropdims reduction should succeed");
            assert_eq!(keep.shape().len(), arr.shape().len());
            assert_eq!(drop.shape().len(), arr.shape().len() - 1);
            assert_eq!(keep.values(), drop.values());
        }
    }

    #[test]
    fn reduce_prod_axis_none() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");

        let out = arr.reduce_prod(None, false).expect("prod all");
        assert_eq!(out.shape(), &[]);
        assert!((out.values()[0] - 720.0).abs() < 1e-10);
    }

    #[test]
    fn reduce_prod_axis_one_keepdims() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");

        let out = arr.reduce_prod(Some(1), true).expect("prod axis=1");
        assert_eq!(out.shape(), &[2, 1]);
        assert!((out.values()[0] - 6.0).abs() < 1e-10);
        assert!((out.values()[1] - 120.0).abs() < 1e-10);
    }

    #[test]
    fn reduce_prod_promotes_i32_to_i64() {
        let arr = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::I32).expect("arr");
        let out = arr.reduce_prod(None, false).expect("prod");
        assert_eq!(out.dtype(), DType::I64);
    }

    #[test]
    fn reduce_min_axis_none() {
        let arr = UFuncArray::new(vec![2, 3], vec![3.0, 1.0, 4.0, 1.5, 5.0, 2.0], DType::F64)
            .expect("arr");

        let out = arr.reduce_min(None, false).expect("min all");
        assert_eq!(out.shape(), &[]);
        assert!((out.values()[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn reduce_min_axis_zero() {
        let arr = UFuncArray::new(vec![2, 3], vec![3.0, 1.0, 4.0, 1.5, 5.0, 2.0], DType::F64)
            .expect("arr");

        let out = arr.reduce_min(Some(0), false).expect("min axis=0");
        assert_eq!(out.shape(), &[3]);
        assert!((out.values()[0] - 1.5).abs() < 1e-10);
        assert!((out.values()[1] - 1.0).abs() < 1e-10);
        assert!((out.values()[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn reduce_min_preserves_dtype() {
        let arr = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::I32).expect("arr");
        let out = arr.reduce_min(None, false).expect("min");
        assert_eq!(out.dtype(), DType::I32);
    }

    #[test]
    fn reduce_max_axis_none() {
        let arr = UFuncArray::new(vec![2, 3], vec![3.0, 1.0, 4.0, 1.5, 5.0, 2.0], DType::F64)
            .expect("arr");

        let out = arr.reduce_max(None, false).expect("max all");
        assert_eq!(out.shape(), &[]);
        assert!((out.values()[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn reduce_max_axis_one_keepdims() {
        let arr = UFuncArray::new(vec![2, 3], vec![3.0, 1.0, 4.0, 1.5, 5.0, 2.0], DType::F64)
            .expect("arr");

        let out = arr.reduce_max(Some(1), true).expect("max axis=1");
        assert_eq!(out.shape(), &[2, 1]);
        assert!((out.values()[0] - 4.0).abs() < 1e-10);
        assert!((out.values()[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn reduce_max_preserves_dtype() {
        let arr = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::Bool).expect("arr");
        let out = arr.reduce_max(None, false).expect("max");
        assert_eq!(out.dtype(), DType::Bool);
    }

    #[test]
    fn reduce_mean_axis_none() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");

        let out = arr.reduce_mean(None, false).expect("mean all");
        assert_eq!(out.shape(), &[]);
        assert!((out.values()[0] - 3.5).abs() < 1e-10);
    }

    #[test]
    fn reduce_mean_axis_one() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");

        let out = arr.reduce_mean(Some(1), false).expect("mean axis=1");
        assert_eq!(out.shape(), &[2]);
        assert!((out.values()[0] - 2.0).abs() < 1e-10);
        assert!((out.values()[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn reduce_mean_promotes_i32_to_f64() {
        let arr = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::I32).expect("arr");
        let out = arr.reduce_mean(None, false).expect("mean");
        assert_eq!(out.dtype(), DType::F64);
    }

    #[test]
    fn reduce_mean_preserves_f32_dtype() {
        let arr = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F32).expect("arr");
        let out = arr.reduce_mean(None, false).expect("mean");
        assert_eq!(out.dtype(), DType::F32);
    }

    #[test]
    fn reduce_mean_keepdims_shape_contract() {
        let arr = UFuncArray::new(vec![2, 3, 4], (1..=24).map(f64::from).collect(), DType::F64)
            .expect("arr");

        for axis in 0_isize..3_isize {
            let keep = arr
                .reduce_mean(Some(axis), true)
                .expect("keepdims reduction");
            let drop = arr
                .reduce_mean(Some(axis), false)
                .expect("dropdims reduction");
            assert_eq!(keep.shape().len(), arr.shape().len());
            assert_eq!(drop.shape().len(), arr.shape().len() - 1);
            assert_eq!(keep.values(), drop.values());
        }
    }

    #[test]
    fn unary_abs_mixed_signs() {
        let arr = UFuncArray::new(
            vec![2, 3],
            vec![-3.0, 1.0, -4.0, 1.5, -5.0, 0.0],
            DType::F64,
        )
        .expect("arr");

        let out = arr.elementwise_unary(UnaryOp::Abs);
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.values(), &[3.0, 1.0, 4.0, 1.5, 5.0, 0.0]);
        assert_eq!(out.dtype(), DType::F64);
    }

    #[test]
    fn unary_negative() {
        let arr = UFuncArray::new(vec![4], vec![1.0, -2.0, 0.0, 3.5], DType::F64).expect("arr");

        let out = arr.elementwise_unary(UnaryOp::Negative);
        assert_eq!(out.values(), &[-1.0, 2.0, 0.0, -3.5]);
    }

    #[test]
    fn unary_sign() {
        let arr = UFuncArray::new(vec![5], vec![-3.0, 0.0, 4.0, f64::NAN, -0.0], DType::F64)
            .expect("arr");

        let out = arr.elementwise_unary(UnaryOp::Sign);
        assert_eq!(out.values()[0], -1.0);
        assert_eq!(out.values()[1], 0.0);
        assert_eq!(out.values()[2], 1.0);
        assert!(out.values()[3].is_nan());
        assert_eq!(out.values()[4], 0.0);
    }

    #[test]
    fn unary_sqrt() {
        let arr = UFuncArray::new(vec![4], vec![0.0, 1.0, 4.0, 9.0], DType::F64).expect("arr");

        let out = arr.elementwise_unary(UnaryOp::Sqrt);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - 1.0).abs() < 1e-10);
        assert!((out.values()[2] - 2.0).abs() < 1e-10);
        assert!((out.values()[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn unary_square() {
        let arr = UFuncArray::new(vec![4], vec![-2.0, 0.0, 3.0, -1.5], DType::F64).expect("arr");

        let out = arr.elementwise_unary(UnaryOp::Square);
        assert!((out.values()[0] - 4.0).abs() < 1e-10);
        assert!((out.values()[1] - 0.0).abs() < 1e-10);
        assert!((out.values()[2] - 9.0).abs() < 1e-10);
        assert!((out.values()[3] - 2.25).abs() < 1e-10);
    }

    #[test]
    fn unary_exp() {
        let arr = UFuncArray::new(vec![3], vec![0.0, 1.0, -1.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Exp);
        assert!((out.values()[0] - 1.0).abs() < 1e-10);
        assert!((out.values()[1] - std::f64::consts::E).abs() < 1e-10);
        assert!((out.values()[2] - 1.0 / std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn unary_log() {
        let arr = UFuncArray::new(vec![3], vec![1.0, std::f64::consts::E, 10.0], DType::F64)
            .expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Log);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - 1.0).abs() < 1e-10);
        assert!((out.values()[2] - 10.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn unary_log2() {
        let arr = UFuncArray::new(vec![4], vec![1.0, 2.0, 4.0, 8.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Log2);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - 1.0).abs() < 1e-10);
        assert!((out.values()[2] - 2.0).abs() < 1e-10);
        assert!((out.values()[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn unary_log10() {
        let arr = UFuncArray::new(vec![3], vec![1.0, 10.0, 100.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Log10);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - 1.0).abs() < 1e-10);
        assert!((out.values()[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn unary_sin_cos_identity() {
        let arr = UFuncArray::new(vec![4], vec![0.0, 0.5, 1.0, 1.5], DType::F64).expect("arr");

        let s = arr.elementwise_unary(UnaryOp::Sin);
        let c = arr.elementwise_unary(UnaryOp::Cos);
        for i in 0..4 {
            let identity = s.values()[i] * s.values()[i] + c.values()[i] * c.values()[i];
            assert!(
                (identity - 1.0).abs() < 1e-10,
                "sin^2 + cos^2 != 1 at index {i}"
            );
        }
    }

    #[test]
    fn unary_tan() {
        let arr = UFuncArray::new(vec![3], vec![0.0, 0.25, 0.5], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Tan);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - 0.25_f64.tan()).abs() < 1e-10);
        assert!((out.values()[2] - 0.5_f64.tan()).abs() < 1e-10);
    }

    #[test]
    fn unary_preserves_dtype() {
        let arr = UFuncArray::new(vec![2], vec![1.0, 4.0], DType::F32).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Sqrt);
        assert_eq!(out.dtype(), DType::F32);
    }

    #[test]
    fn binary_power() {
        let lhs = UFuncArray::new(vec![3], vec![2.0, 3.0, 4.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![3], vec![3.0, 2.0, 0.5], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Power)
            .expect("power");
        assert!((out.values()[0] - 8.0).abs() < 1e-10);
        assert!((out.values()[1] - 9.0).abs() < 1e-10);
        assert!((out.values()[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn binary_remainder_sign_follows_divisor() {
        let lhs = UFuncArray::new(vec![4], vec![7.0, -7.0, 7.0, -7.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![4], vec![3.0, 3.0, -3.0, -3.0], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Remainder)
            .expect("remainder");
        assert!((out.values()[0] - 1.0).abs() < 1e-10);
        assert!((out.values()[1] - 2.0).abs() < 1e-10);
        assert!((out.values()[2] - (-2.0)).abs() < 1e-10);
        assert!((out.values()[3] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn binary_minimum() {
        let lhs = UFuncArray::new(vec![3], vec![1.0, 5.0, 3.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![3], vec![4.0, 2.0, 6.0], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Minimum)
            .expect("minimum");
        assert_eq!(out.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn binary_maximum() {
        let lhs = UFuncArray::new(vec![3], vec![1.0, 5.0, 3.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![3], vec![4.0, 2.0, 6.0], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Maximum)
            .expect("maximum");
        assert_eq!(out.values(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn binary_minimum_nan_propagation() {
        let lhs = UFuncArray::new(vec![3], vec![1.0, f64::NAN, 3.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![3], vec![f64::NAN, 2.0, f64::NAN], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Minimum)
            .expect("minimum");
        assert!(out.values()[0].is_nan());
        assert!(out.values()[1].is_nan());
        assert!(out.values()[2].is_nan());
    }

    #[test]
    fn unary_floor() {
        let arr =
            UFuncArray::new(vec![5], vec![1.7, -1.7, 2.5, -2.5, 0.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Floor);
        assert_eq!(out.values(), &[1.0, -2.0, 2.0, -3.0, 0.0]);
    }

    #[test]
    fn unary_ceil() {
        let arr =
            UFuncArray::new(vec![5], vec![1.7, -1.7, 2.5, -2.5, 0.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Ceil);
        assert_eq!(out.values(), &[2.0, -1.0, 3.0, -2.0, 0.0]);
    }

    #[test]
    fn unary_round_bankers() {
        let arr =
            UFuncArray::new(vec![6], vec![0.5, 1.5, 2.5, 3.5, 4.5, -0.5], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Round);
        // Banker's rounding: 0.5->0, 1.5->2, 2.5->2, 3.5->4, 4.5->4, -0.5->0
        assert_eq!(out.values(), &[0.0, 2.0, 2.0, 4.0, 4.0, 0.0]);
    }

    #[test]
    fn unary_reciprocal() {
        let arr = UFuncArray::new(vec![4], vec![1.0, 2.0, 4.0, 0.5], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Reciprocal);
        assert!((out.values()[0] - 1.0).abs() < 1e-10);
        assert!((out.values()[1] - 0.5).abs() < 1e-10);
        assert!((out.values()[2] - 0.25).abs() < 1e-10);
        assert!((out.values()[3] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn unary_sinh() {
        let arr = UFuncArray::new(vec![3], vec![0.0, 1.0, -1.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Sinh);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - 1.0_f64.sinh()).abs() < 1e-10);
        assert!((out.values()[2] - (-1.0_f64).sinh()).abs() < 1e-10);
    }

    #[test]
    fn unary_cosh() {
        let arr = UFuncArray::new(vec![3], vec![0.0, 1.0, -1.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Cosh);
        assert!((out.values()[0] - 1.0).abs() < 1e-10);
        assert!((out.values()[1] - 1.0_f64.cosh()).abs() < 1e-10);
        assert!((out.values()[2] - 1.0_f64.cosh()).abs() < 1e-10); // cosh is even
    }

    #[test]
    fn unary_tanh() {
        let arr = UFuncArray::new(vec![3], vec![0.0, 1.0, -1.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Tanh);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - 1.0_f64.tanh()).abs() < 1e-10);
        assert!((out.values()[2] - (-1.0_f64).tanh()).abs() < 1e-10);
    }

    #[test]
    fn unary_arcsin() {
        let arr = UFuncArray::new(vec![3], vec![0.0, 0.5, 1.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Arcsin);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - 0.5_f64.asin()).abs() < 1e-10);
        assert!((out.values()[2] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn unary_arccos() {
        let arr = UFuncArray::new(vec![3], vec![1.0, 0.5, 0.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Arccos);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - 0.5_f64.acos()).abs() < 1e-10);
        assert!((out.values()[2] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn unary_arctan() {
        let arr = UFuncArray::new(vec![3], vec![0.0, 1.0, -1.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Arctan);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
        assert!((out.values()[2] - (-std::f64::consts::FRAC_PI_4)).abs() < 1e-10);
    }

    #[test]
    fn binary_arctan2() {
        let lhs = UFuncArray::new(vec![4], vec![1.0, -1.0, -1.0, 1.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![4], vec![1.0, 1.0, -1.0, -1.0], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Arctan2)
            .expect("arctan2");
        assert!((out.values()[0] - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
        assert!((out.values()[1] - (-std::f64::consts::FRAC_PI_4)).abs() < 1e-10);
    }

    #[test]
    fn unary_cbrt() {
        let arr = UFuncArray::new(vec![4], vec![8.0, -27.0, 0.0, 1.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Cbrt);
        assert!((out.values()[0] - 2.0).abs() < 1e-10);
        assert!((out.values()[1] - (-3.0)).abs() < 1e-10);
        assert!((out.values()[2] - 0.0).abs() < 1e-10);
        assert!((out.values()[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn unary_expm1() {
        let arr = UFuncArray::new(vec![3], vec![0.0, 1.0, -1.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Expm1);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - (std::f64::consts::E - 1.0)).abs() < 1e-10);
        assert!((out.values()[2] - ((-1.0_f64).exp() - 1.0)).abs() < 1e-10);
    }

    #[test]
    fn unary_log1p() {
        let arr = UFuncArray::new(vec![3], vec![0.0, 1.0, -0.5], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Log1p);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - 2.0_f64.ln()).abs() < 1e-10);
        assert!((out.values()[2] - 0.5_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn unary_degrees() {
        let arr = UFuncArray::new(
            vec![4],
            vec![
                0.0,
                std::f64::consts::PI,
                std::f64::consts::FRAC_PI_2,
                std::f64::consts::FRAC_PI_4,
            ],
            DType::F64,
        )
        .expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Degrees);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - 180.0).abs() < 1e-10);
        assert!((out.values()[2] - 90.0).abs() < 1e-10);
        assert!((out.values()[3] - 45.0).abs() < 1e-10);
    }

    #[test]
    fn unary_radians() {
        let arr = UFuncArray::new(vec![4], vec![0.0, 180.0, 90.0, 45.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Radians);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - std::f64::consts::PI).abs() < 1e-10);
        assert!((out.values()[2] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
        assert!((out.values()[3] - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
    }

    #[test]
    fn unary_rint_bankers() {
        let arr = UFuncArray::new(
            vec![8],
            vec![0.5, 1.5, 2.5, 3.5, 4.5, -0.5, 1.7, -1.7],
            DType::F64,
        )
        .expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Rint);
        // Banker's rounding: 0.5->0, 1.5->2, 2.5->2, 3.5->4, 4.5->4, -0.5->0
        assert_eq!(out.values()[0], 0.0);
        assert_eq!(out.values()[1], 2.0);
        assert_eq!(out.values()[2], 2.0);
        assert_eq!(out.values()[3], 4.0);
        assert_eq!(out.values()[4], 4.0);
        assert_eq!(out.values()[5], 0.0);
        assert!((out.values()[6] - 2.0).abs() < 1e-10);
        assert!((out.values()[7] - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn unary_trunc() {
        let arr =
            UFuncArray::new(vec![5], vec![1.7, -1.7, 2.5, -2.5, 0.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Trunc);
        assert_eq!(out.values(), &[1.0, -1.0, 2.0, -2.0, 0.0]);
    }

    #[test]
    fn binary_fmod_sign_follows_dividend() {
        let lhs = UFuncArray::new(vec![4], vec![7.0, -7.0, 7.0, -7.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![4], vec![3.0, 3.0, -3.0, -3.0], DType::F64).expect("rhs");
        let out = lhs.elementwise_binary(&rhs, BinaryOp::Fmod).expect("fmod");
        // fmod: sign follows dividend (unlike remainder where sign follows divisor)
        assert!((out.values()[0] - 1.0).abs() < 1e-10);
        assert!((out.values()[1] - (-1.0)).abs() < 1e-10);
        assert!((out.values()[2] - 1.0).abs() < 1e-10);
        assert!((out.values()[3] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn binary_copysign() {
        let lhs = UFuncArray::new(vec![4], vec![1.0, -1.0, 3.0, -3.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![4], vec![-1.0, 1.0, -1.0, 1.0], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Copysign)
            .expect("copysign");
        assert_eq!(out.values(), &[-1.0, 1.0, -3.0, 3.0]);
    }

    #[test]
    fn binary_fmax() {
        let lhs = UFuncArray::new(vec![3], vec![1.0, 5.0, 3.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![3], vec![4.0, 2.0, 6.0], DType::F64).expect("rhs");
        let out = lhs.elementwise_binary(&rhs, BinaryOp::Fmax).expect("fmax");
        assert_eq!(out.values(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn binary_fmin() {
        let lhs = UFuncArray::new(vec![3], vec![1.0, 5.0, 3.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![3], vec![4.0, 2.0, 6.0], DType::F64).expect("rhs");
        let out = lhs.elementwise_binary(&rhs, BinaryOp::Fmin).expect("fmin");
        assert_eq!(out.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn binary_fmax_nan_ignoring() {
        let lhs = UFuncArray::new(vec![3], vec![1.0, f64::NAN, 3.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![3], vec![f64::NAN, 2.0, f64::NAN], DType::F64).expect("rhs");
        let out = lhs.elementwise_binary(&rhs, BinaryOp::Fmax).expect("fmax");
        assert_eq!(out.values()[0], 1.0);
        assert_eq!(out.values()[1], 2.0);
        assert_eq!(out.values()[2], 3.0);
    }

    #[test]
    fn binary_fmin_nan_ignoring() {
        let lhs = UFuncArray::new(vec![3], vec![1.0, f64::NAN, 3.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![3], vec![f64::NAN, 2.0, f64::NAN], DType::F64).expect("rhs");
        let out = lhs.elementwise_binary(&rhs, BinaryOp::Fmin).expect("fmin");
        assert_eq!(out.values()[0], 1.0);
        assert_eq!(out.values()[1], 2.0);
        assert_eq!(out.values()[2], 3.0);
    }

    #[test]
    fn unary_positive() {
        let arr = UFuncArray::new(vec![4], vec![-3.0, 0.0, 5.0, -1.5], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Positive);
        assert_eq!(out.values(), &[-3.0, 0.0, 5.0, -1.5]);
    }

    #[test]
    fn unary_spacing() {
        let arr = UFuncArray::new(vec![3], vec![1.0, 0.0, -1.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Spacing);
        // spacing(1.0) = machine epsilon for f64
        assert!((out.values()[0] - f64::EPSILON).abs() < 1e-30);
        // spacing(0.0) = smallest positive subnormal
        assert!(out.values()[1] > 0.0 && out.values()[1] < 1e-300);
        // spacing(-1.0) = same as spacing(1.0) since it uses abs
        assert!((out.values()[2] - f64::EPSILON).abs() < 1e-30);
    }

    #[test]
    fn binary_heaviside() {
        let lhs = UFuncArray::new(vec![3], vec![-1.5, 0.0, 2.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![3], vec![0.5, 0.5, 0.5], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Heaviside)
            .expect("heaviside");
        assert_eq!(out.values(), &[0.0, 0.5, 1.0]);
    }

    #[test]
    fn binary_nextafter() {
        let lhs = UFuncArray::new(vec![2], vec![1.0, 0.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![2], vec![2.0, -1.0], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Nextafter)
            .expect("nextafter");
        // nextafter(1.0, 2.0) should be slightly > 1.0
        assert!(out.values()[0] > 1.0);
        assert!((out.values()[0] - 1.0).abs() < 1e-15);
        // nextafter(0.0, -1.0) should be smallest negative subnormal
        assert!(out.values()[1] < 0.0);
        assert!(out.values()[1] > -1e-300);
    }

    #[test]
    fn unary_logical_not() {
        let arr =
            UFuncArray::new(vec![5], vec![0.0, 1.0, -1.0, 0.0, 3.5], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::LogicalNot);
        assert_eq!(out.values(), &[1.0, 0.0, 0.0, 1.0, 0.0]);
        assert_eq!(out.dtype(), DType::Bool);
    }

    #[test]
    fn unary_logical_not_nan_is_truthy() {
        let arr = UFuncArray::new(vec![2], vec![f64::NAN, 0.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::LogicalNot);
        // NaN != 0.0 is true in IEEE 754, so NaN is truthy -> logical_not returns 0.0
        assert_eq!(out.values()[0], 0.0);
        assert_eq!(out.values()[1], 1.0);
        assert_eq!(out.dtype(), DType::Bool);
    }

    #[test]
    fn binary_logical_and() {
        let lhs = UFuncArray::new(vec![4], vec![0.0, 0.0, 1.0, 1.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![4], vec![0.0, 1.0, 0.0, 1.0], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::LogicalAnd)
            .expect("logical_and");
        assert_eq!(out.values(), &[0.0, 0.0, 0.0, 1.0]);
        assert_eq!(out.dtype(), DType::Bool);
    }

    #[test]
    fn binary_logical_or() {
        let lhs = UFuncArray::new(vec![4], vec![0.0, 0.0, 1.0, 1.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![4], vec![0.0, 1.0, 0.0, 1.0], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::LogicalOr)
            .expect("logical_or");
        assert_eq!(out.values(), &[0.0, 1.0, 1.0, 1.0]);
        assert_eq!(out.dtype(), DType::Bool);
    }

    #[test]
    fn binary_logical_xor() {
        let lhs = UFuncArray::new(vec![4], vec![0.0, 0.0, 1.0, 1.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![4], vec![0.0, 1.0, 0.0, 1.0], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::LogicalXor)
            .expect("logical_xor");
        assert_eq!(out.values(), &[0.0, 1.0, 1.0, 0.0]);
        assert_eq!(out.dtype(), DType::Bool);
    }

    #[test]
    fn binary_equal() {
        let lhs = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![4], vec![1.0, 3.0, 3.0, 5.0], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Equal)
            .expect("equal");
        assert_eq!(out.values(), &[1.0, 0.0, 1.0, 0.0]);
        assert_eq!(out.dtype(), DType::Bool);
    }

    #[test]
    fn binary_equal_nan_semantics() {
        let lhs = UFuncArray::new(vec![2], vec![f64::NAN, f64::NAN], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![2], vec![f64::NAN, 1.0], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Equal)
            .expect("equal");
        // IEEE 754: NaN == NaN is false, NaN == 1.0 is false
        assert_eq!(out.values(), &[0.0, 0.0]);
    }

    #[test]
    fn binary_not_equal() {
        let lhs = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![4], vec![1.0, 3.0, 3.0, 5.0], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::NotEqual)
            .expect("not_equal");
        assert_eq!(out.values(), &[0.0, 1.0, 0.0, 1.0]);
        assert_eq!(out.dtype(), DType::Bool);
    }

    #[test]
    fn binary_not_equal_nan_semantics() {
        let lhs = UFuncArray::new(vec![2], vec![f64::NAN, f64::NAN], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![2], vec![f64::NAN, 1.0], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::NotEqual)
            .expect("not_equal");
        // IEEE 754: NaN != NaN is true, NaN != 1.0 is true
        assert_eq!(out.values(), &[1.0, 1.0]);
    }

    #[test]
    fn binary_less() {
        let lhs = UFuncArray::new(vec![4], vec![1.0, 3.0, 2.0, 4.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![4], vec![2.0, 2.0, 3.0, 4.0], DType::F64).expect("rhs");
        let out = lhs.elementwise_binary(&rhs, BinaryOp::Less).expect("less");
        assert_eq!(out.values(), &[1.0, 0.0, 1.0, 0.0]);
        assert_eq!(out.dtype(), DType::Bool);
    }

    #[test]
    fn binary_less_equal() {
        let lhs = UFuncArray::new(vec![4], vec![1.0, 3.0, 2.0, 4.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![4], vec![2.0, 2.0, 3.0, 4.0], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::LessEqual)
            .expect("less_equal");
        assert_eq!(out.values(), &[1.0, 0.0, 1.0, 1.0]);
        assert_eq!(out.dtype(), DType::Bool);
    }

    #[test]
    fn binary_greater() {
        let lhs = UFuncArray::new(vec![4], vec![1.0, 3.0, 2.0, 4.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![4], vec![2.0, 2.0, 3.0, 4.0], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Greater)
            .expect("greater");
        assert_eq!(out.values(), &[0.0, 1.0, 0.0, 0.0]);
        assert_eq!(out.dtype(), DType::Bool);
    }

    #[test]
    fn binary_greater_equal() {
        let lhs = UFuncArray::new(vec![4], vec![1.0, 3.0, 2.0, 4.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![4], vec![2.0, 2.0, 3.0, 4.0], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::GreaterEqual)
            .expect("greater_equal");
        assert_eq!(out.values(), &[0.0, 1.0, 0.0, 1.0]);
        assert_eq!(out.dtype(), DType::Bool);
    }

    #[test]
    fn unary_isnan() {
        let arr =
            UFuncArray::new(vec![4], vec![0.0, f64::NAN, 1.0, f64::NAN], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Isnan);
        assert_eq!(out.values(), &[0.0, 1.0, 0.0, 1.0]);
        assert_eq!(out.dtype(), DType::Bool);
    }

    #[test]
    fn unary_isinf() {
        let arr = UFuncArray::new(
            vec![5],
            vec![0.0, f64::INFINITY, f64::NEG_INFINITY, 1.0, f64::NAN],
            DType::F64,
        )
        .expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Isinf);
        assert_eq!(out.values(), &[0.0, 1.0, 1.0, 0.0, 0.0]);
        assert_eq!(out.dtype(), DType::Bool);
    }

    #[test]
    fn unary_isfinite() {
        let arr = UFuncArray::new(
            vec![5],
            vec![0.0, 1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
            DType::F64,
        )
        .expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Isfinite);
        assert_eq!(out.values(), &[1.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(out.dtype(), DType::Bool);
    }

    #[test]
    fn unary_signbit() {
        let arr = UFuncArray::new(
            vec![5],
            vec![1.0, -1.0, 0.0, f64::INFINITY, f64::NEG_INFINITY],
            DType::F64,
        )
        .expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Signbit);
        assert_eq!(out.values(), &[0.0, 1.0, 0.0, 0.0, 1.0]);
        assert_eq!(out.dtype(), DType::Bool);
    }

    #[test]
    fn binary_hypot() {
        let lhs = UFuncArray::new(vec![3], vec![3.0, 5.0, 1.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![3], vec![4.0, 12.0, 0.0], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Hypot)
            .expect("hypot");
        assert!((out.values()[0] - 5.0).abs() < 1e-10);
        assert!((out.values()[1] - 13.0).abs() < 1e-10);
        assert!((out.values()[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn binary_logaddexp() {
        let lhs = UFuncArray::new(vec![2], vec![1.0, 0.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![2], vec![2.0, 0.0], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Logaddexp)
            .expect("logaddexp");
        // logaddexp(1, 2) = log(e + e^2) = 2.31326...
        let expected0 = (1.0_f64.exp() + 2.0_f64.exp()).ln();
        assert!((out.values()[0] - expected0).abs() < 1e-10);
        // logaddexp(0, 0) = log(1 + 1) = ln(2)
        assert!((out.values()[1] - 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn binary_logaddexp2() {
        let lhs = UFuncArray::new(vec![2], vec![1.0, 0.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![2], vec![2.0, 0.0], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Logaddexp2)
            .expect("logaddexp2");
        // logaddexp2(1, 2) = log2(2 + 4) = log2(6)
        assert!((out.values()[0] - 6.0_f64.log2()).abs() < 1e-10);
        // logaddexp2(0, 0) = log2(1 + 1) = 1.0
        assert!((out.values()[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn unary_exp2() {
        let arr = UFuncArray::new(vec![4], vec![0.0, 1.0, 2.0, 3.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Exp2);
        assert_eq!(out.values(), &[1.0, 2.0, 4.0, 8.0]);
    }

    #[test]
    fn unary_fabs() {
        let arr = UFuncArray::new(vec![4], vec![-3.0, 0.0, 5.0, -1.5], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Fabs);
        assert_eq!(out.values(), &[3.0, 0.0, 5.0, 1.5]);
    }

    #[test]
    fn unary_arccosh() {
        let arr = UFuncArray::new(vec![3], vec![1.0, 2.0, 10.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Arccosh);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - 2.0_f64.acosh()).abs() < 1e-10);
        assert!((out.values()[2] - 10.0_f64.acosh()).abs() < 1e-10);
    }

    #[test]
    fn unary_arcsinh() {
        let arr = UFuncArray::new(vec![3], vec![0.0, 1.0, -1.0], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Arcsinh);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - 1.0_f64.asinh()).abs() < 1e-10);
        assert!((out.values()[2] - (-1.0_f64).asinh()).abs() < 1e-10);
    }

    #[test]
    fn unary_arctanh() {
        let arr = UFuncArray::new(vec![3], vec![0.0, 0.5, -0.5], DType::F64).expect("arr");
        let out = arr.elementwise_unary(UnaryOp::Arctanh);
        assert!((out.values()[0] - 0.0).abs() < 1e-10);
        assert!((out.values()[1] - 0.5_f64.atanh()).abs() < 1e-10);
        assert!((out.values()[2] - (-0.5_f64).atanh()).abs() < 1e-10);
    }

    #[test]
    fn binary_floor_divide() {
        let lhs = UFuncArray::new(vec![4], vec![7.0, -7.0, 7.0, -7.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![4], vec![3.0, 3.0, -3.0, -3.0], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::FloorDivide)
            .expect("floor_divide");
        assert!((out.values()[0] - 2.0).abs() < 1e-10);
        assert!((out.values()[1] - (-3.0)).abs() < 1e-10);
        assert!((out.values()[2] - (-3.0)).abs() < 1e-10);
        assert!((out.values()[3] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn binary_float_power() {
        let lhs = UFuncArray::new(vec![3], vec![2.0, 3.0, 4.0], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![3], vec![3.0, 2.0, 0.5], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::FloatPower)
            .expect("float_power");
        assert!((out.values()[0] - 8.0).abs() < 1e-10);
        assert!((out.values()[1] - 9.0).abs() < 1e-10);
        assert!((out.values()[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn binary_ldexp() {
        let lhs = UFuncArray::new(vec![4], vec![1.0, 1.5, 2.0, 0.5], DType::F64).expect("lhs");
        let rhs = UFuncArray::new(vec![4], vec![2.0, 3.0, 0.0, 4.0], DType::F64).expect("rhs");
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Ldexp)
            .expect("ldexp");
        // ldexp(1.0, 2) = 4.0, ldexp(1.5, 3) = 12.0, ldexp(2.0, 0) = 2.0, ldexp(0.5, 4) = 8.0
        assert!((out.values()[0] - 4.0).abs() < 1e-10);
        assert!((out.values()[1] - 12.0).abs() < 1e-10);
        assert!((out.values()[2] - 2.0).abs() < 1e-10);
        assert!((out.values()[3] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn comparison_broadcast_produces_bool_dtype() {
        let lhs = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("lhs");
        let rhs = UFuncArray::scalar(3.0, DType::F64);
        let out = lhs
            .elementwise_binary(&rhs, BinaryOp::Equal)
            .expect("broadcast equal");
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.values(), &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(out.dtype(), DType::Bool);
    }

    #[test]
    fn cumsum_axis_none_flattens() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");
        let out = arr.cumsum(None).expect("cumsum axis=None");
        assert_eq!(out.shape(), &[6]);
        assert_eq!(out.values(), &[1.0, 3.0, 6.0, 10.0, 15.0, 21.0]);
    }

    #[test]
    fn cumsum_axis_0_preserves_shape() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");
        let out = arr.cumsum(Some(0)).expect("cumsum axis=0");
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.values(), &[1.0, 2.0, 3.0, 5.0, 7.0, 9.0]);
    }

    #[test]
    fn cumsum_axis_1() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");
        let out = arr.cumsum(Some(1)).expect("cumsum axis=1");
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.values(), &[1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
    }

    #[test]
    fn cumsum_promotes_i32_to_i64() {
        let arr = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::I32).expect("arr");
        let out = arr.cumsum(None).expect("cumsum");
        assert_eq!(out.dtype(), DType::I64);
    }

    #[test]
    fn cumprod_axis_none_flattens() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");
        let out = arr.cumprod(None).expect("cumprod axis=None");
        assert_eq!(out.shape(), &[6]);
        assert_eq!(out.values(), &[1.0, 2.0, 6.0, 24.0, 120.0, 720.0]);
    }

    #[test]
    fn cumprod_axis_1() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");
        let out = arr.cumprod(Some(1)).expect("cumprod axis=1");
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.values(), &[1.0, 2.0, 6.0, 4.0, 20.0, 120.0]);
    }

    #[test]
    fn clip_clamps_values() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 5.0, 3.0, 8.0, 2.0, 7.0], DType::F64)
            .expect("arr");
        let out = arr.clip(2.0, 6.0);
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.values(), &[2.0, 5.0, 3.0, 6.0, 2.0, 6.0]);
        assert_eq!(out.dtype(), DType::F64);
    }

    #[test]
    fn clip_min_only() {
        let arr = UFuncArray::new(vec![6], vec![-3.0, -1.0, 0.0, 1.0, 3.0, 5.0], DType::F64)
            .expect("arr");
        let out = arr.clip(0.0, f64::INFINITY);
        assert_eq!(out.values(), &[0.0, 0.0, 0.0, 1.0, 3.0, 5.0]);
    }

    #[test]
    fn cumsum_3d_axis_2() {
        let arr = UFuncArray::new(
            vec![2, 2, 3],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            DType::F64,
        )
        .expect("arr");
        let out = arr.cumsum(Some(2)).expect("cumsum axis=2");
        assert_eq!(out.shape(), &[2, 2, 3]);
        assert_eq!(
            out.values(),
            &[
                1.0, 3.0, 6.0, 4.0, 9.0, 15.0, 7.0, 15.0, 24.0, 10.0, 21.0, 33.0
            ]
        );
    }

    // ── var / std reduction tests ──────────────────────────────────────

    #[test]
    fn reduce_var_axis_none() {
        // np.var([1,2,3,4]) = 1.25
        let arr = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).expect("arr");
        let out = arr.reduce_var(None, false, 0).expect("var");
        assert_eq!(out.shape(), &[] as &[usize]);
        assert!((out.values()[0] - 1.25).abs() < 1e-12);
    }

    #[test]
    fn reduce_var_ddof_1() {
        // np.var([1,2,3,4], ddof=1) = 1.6666...
        let arr = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).expect("arr");
        let out = arr.reduce_var(None, false, 1).expect("var ddof=1");
        assert!((out.values()[0] - 5.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn reduce_var_axis_one() {
        // np.var([[1,2,3],[4,5,6]], axis=1) = [0.6666..., 0.6666...]
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");
        let out = arr.reduce_var(Some(1), false, 0).expect("var axis=1");
        assert_eq!(out.shape(), &[2]);
        let expected = 2.0 / 3.0;
        assert!((out.values()[0] - expected).abs() < 1e-12);
        assert!((out.values()[1] - expected).abs() < 1e-12);
    }

    #[test]
    fn reduce_var_keepdims() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");
        let out = arr.reduce_var(Some(1), true, 0).expect("var keepdims");
        assert_eq!(out.shape(), &[2, 1]);
    }

    #[test]
    fn reduce_var_promotes_i32_to_f64() {
        let arr = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::I32).expect("arr");
        let out = arr.reduce_var(None, false, 0).expect("var");
        assert_eq!(out.dtype(), DType::F64);
    }

    #[test]
    fn reduce_std_axis_none() {
        // np.std([1,2,3,4]) = sqrt(1.25) ≈ 1.118033988749895
        let arr = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).expect("arr");
        let out = arr.reduce_std(None, false, 0).expect("std");
        assert!((out.values()[0] - 1.25_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn reduce_std_axis_zero() {
        // np.std([[1,5],[2,6]], axis=0) = [0.5, 0.5]
        let arr = UFuncArray::new(vec![2, 2], vec![1.0, 5.0, 2.0, 6.0], DType::F64).expect("arr");
        let out = arr.reduce_std(Some(0), false, 0).expect("std axis=0");
        assert_eq!(out.shape(), &[2]);
        assert!((out.values()[0] - 0.5).abs() < 1e-12);
        assert!((out.values()[1] - 0.5).abs() < 1e-12);
    }

    // ── argmin / argmax reduction tests ─────────────────────────────────

    #[test]
    fn reduce_argmin_axis_none() {
        let arr = UFuncArray::new(vec![2, 3], vec![5.0, 1.0, 3.0, 2.0, 4.0, 0.0], DType::F64)
            .expect("arr");
        let out = arr.reduce_argmin(None).expect("argmin");
        assert_eq!(out.shape(), &[] as &[usize]);
        assert_eq!(out.values()[0], 5.0); // flat index of 0.0
        assert_eq!(out.dtype(), DType::I64);
    }

    #[test]
    fn reduce_argmin_axis_zero() {
        // np.argmin([[5,1],[3,2]], axis=0) = [1, 0]
        let arr = UFuncArray::new(vec![2, 2], vec![5.0, 1.0, 3.0, 2.0], DType::F64).expect("arr");
        let out = arr.reduce_argmin(Some(0)).expect("argmin axis=0");
        assert_eq!(out.shape(), &[2]);
        assert_eq!(out.values(), &[1.0, 0.0]);
    }

    #[test]
    fn reduce_argmin_axis_one() {
        // np.argmin([[5,1,3],[2,4,0]], axis=1) = [1, 2]
        let arr = UFuncArray::new(vec![2, 3], vec![5.0, 1.0, 3.0, 2.0, 4.0, 0.0], DType::F64)
            .expect("arr");
        let out = arr.reduce_argmin(Some(1)).expect("argmin axis=1");
        assert_eq!(out.shape(), &[2]);
        assert_eq!(out.values(), &[1.0, 2.0]);
    }

    #[test]
    fn reduce_argmax_axis_none() {
        let arr = UFuncArray::new(vec![2, 3], vec![5.0, 1.0, 3.0, 2.0, 4.0, 9.0], DType::F64)
            .expect("arr");
        let out = arr.reduce_argmax(None).expect("argmax");
        assert_eq!(out.shape(), &[] as &[usize]);
        assert_eq!(out.values()[0], 5.0); // flat index of 9.0
        assert_eq!(out.dtype(), DType::I64);
    }

    #[test]
    fn reduce_argmax_axis_zero() {
        // np.argmax([[5,1],[3,2]], axis=0) = [0, 1]
        let arr = UFuncArray::new(vec![2, 2], vec![5.0, 1.0, 3.0, 2.0], DType::F64).expect("arr");
        let out = arr.reduce_argmax(Some(0)).expect("argmax axis=0");
        assert_eq!(out.shape(), &[2]);
        assert_eq!(out.values(), &[0.0, 1.0]);
    }

    #[test]
    fn reduce_argmax_axis_one() {
        // np.argmax([[5,1,3],[2,4,9]], axis=1) = [0, 2]
        let arr = UFuncArray::new(vec![2, 3], vec![5.0, 1.0, 3.0, 2.0, 4.0, 9.0], DType::F64)
            .expect("arr");
        let out = arr.reduce_argmax(Some(1)).expect("argmax axis=1");
        assert_eq!(out.shape(), &[2]);
        assert_eq!(out.values(), &[0.0, 2.0]);
    }

    #[test]
    fn reduce_argmax_3d_axis_1() {
        // 2x3x2 array, argmax along axis=1
        let arr = UFuncArray::new(
            vec![2, 3, 2],
            vec![
                1.0, 2.0, 5.0, 4.0, 3.0, 6.0, // first outer slice
                9.0, 8.0, 7.0, 10.0, 11.0, 12.0, // second outer slice
            ],
            DType::F64,
        )
        .expect("arr");
        let out = arr.reduce_argmax(Some(1)).expect("argmax axis=1");
        assert_eq!(out.shape(), &[2, 2]);
        // first outer: col 0 max is 5.0 at row 1, col 1 max is 6.0 at row 2
        // second outer: col 0 max is 11.0 at row 2, col 1 max is 12.0 at row 2
        assert_eq!(out.values(), &[1.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn reduce_argmin_argmax_flat_select_first_nan_index() {
        let arr = UFuncArray::new(vec![3], vec![1.0, f64::NAN, 2.0], DType::F64).expect("arr");
        let argmin = arr.reduce_argmin(None).expect("argmin");
        let argmax = arr.reduce_argmax(None).expect("argmax");
        assert_eq!(argmin.values(), &[1.0]);
        assert_eq!(argmax.values(), &[1.0]);
    }

    #[test]
    fn reduce_argmin_argmax_accept_negative_axis() {
        let arr = UFuncArray::new(vec![2, 3], vec![5.0, 1.0, 3.0, 2.0, 4.0, 9.0], DType::F64)
            .expect("arr");
        let argmin = arr.reduce_argmin(Some(-1)).expect("argmin axis=-1");
        let argmax = arr.reduce_argmax(Some(-1)).expect("argmax axis=-1");
        assert_eq!(argmin.shape(), &[2]);
        assert_eq!(argmin.values(), &[1.0, 0.0]);
        assert_eq!(argmax.shape(), &[2]);
        assert_eq!(argmax.values(), &[0.0, 2.0]);
    }

    #[test]
    fn reduce_argmin_rejects_empty_flattened_input() {
        let arr = UFuncArray::new(vec![0], Vec::new(), DType::F64).expect("arr");
        let err = arr
            .reduce_argmin(None)
            .expect_err("argmin on empty should fail");
        assert!(matches!(err, UFuncError::EmptyReduction { op: "argmin" }));
        assert_eq!(err.reason_code(), "ufunc_reduce_axis_contract");
    }

    #[test]
    fn reduce_argmax_rejects_zero_length_reduction_axis() {
        let arr = UFuncArray::new(vec![2, 0], Vec::new(), DType::F64).expect("arr");
        let err = arr
            .reduce_argmax(Some(1))
            .expect_err("argmax on zero-size axis should fail");
        assert!(matches!(err, UFuncError::EmptyReduction { op: "argmax" }));
        assert_eq!(err.reason_code(), "ufunc_reduce_axis_contract");
    }

    #[test]
    fn reduce_argmin_allows_empty_outer_extent_when_axis_has_length() {
        let arr = UFuncArray::new(vec![0, 2], Vec::new(), DType::F64).expect("arr");
        let out = arr.reduce_argmin(Some(1)).expect("argmin axis=1");
        assert_eq!(out.shape(), &[0]);
        assert!(out.values().is_empty());
    }

    // ── bitwise operation tests ─────────────────────────────────────────

    #[test]
    fn binary_bitwise_and() {
        // np.bitwise_and(12, 10) = 8
        let a = UFuncArray::new(vec![3], vec![12.0, 255.0, 7.0], DType::I64).expect("a");
        let b = UFuncArray::new(vec![3], vec![10.0, 15.0, 3.0], DType::I64).expect("b");
        let out = a.elementwise_binary(&b, BinaryOp::BitwiseAnd).expect("ok");
        assert_eq!(out.values(), &[8.0, 15.0, 3.0]);
    }

    #[test]
    fn binary_bitwise_or() {
        // np.bitwise_or(12, 10) = 14
        let a = UFuncArray::new(vec![2], vec![12.0, 5.0], DType::I64).expect("a");
        let b = UFuncArray::new(vec![2], vec![10.0, 3.0], DType::I64).expect("b");
        let out = a.elementwise_binary(&b, BinaryOp::BitwiseOr).expect("ok");
        assert_eq!(out.values(), &[14.0, 7.0]);
    }

    #[test]
    fn binary_bitwise_xor() {
        // np.bitwise_xor(12, 10) = 6
        let a = UFuncArray::new(vec![2], vec![12.0, 255.0], DType::I64).expect("a");
        let b = UFuncArray::new(vec![2], vec![10.0, 255.0], DType::I64).expect("b");
        let out = a.elementwise_binary(&b, BinaryOp::BitwiseXor).expect("ok");
        assert_eq!(out.values(), &[6.0, 0.0]);
    }

    #[test]
    fn binary_left_shift() {
        // np.left_shift(1, 4) = 16
        let a = UFuncArray::new(vec![2], vec![1.0, 5.0], DType::I64).expect("a");
        let b = UFuncArray::new(vec![2], vec![4.0, 2.0], DType::I64).expect("b");
        let out = a.elementwise_binary(&b, BinaryOp::LeftShift).expect("ok");
        assert_eq!(out.values(), &[16.0, 20.0]);
    }

    #[test]
    fn binary_right_shift() {
        // np.right_shift(16, 4) = 1
        let a = UFuncArray::new(vec![2], vec![16.0, 20.0], DType::I64).expect("a");
        let b = UFuncArray::new(vec![2], vec![4.0, 2.0], DType::I64).expect("b");
        let out = a.elementwise_binary(&b, BinaryOp::RightShift).expect("ok");
        assert_eq!(out.values(), &[1.0, 5.0]);
    }

    #[test]
    fn unary_invert() {
        // np.invert(0) = -1, np.invert(1) = -2, np.invert(-1) = 0
        let a = UFuncArray::new(vec![3], vec![0.0, 1.0, -1.0], DType::I64).expect("a");
        let out = a.elementwise_unary(UnaryOp::Invert);
        assert_eq!(out.values(), &[-1.0, -2.0, 0.0]);
    }

    #[test]
    fn bitwise_and_broadcast() {
        // Scalar broadcast: np.bitwise_and([12, 7], 3) = [0, 3]
        let a = UFuncArray::new(vec![2], vec![12.0, 7.0], DType::I64).expect("a");
        let b = UFuncArray::new(vec![1], vec![3.0], DType::I64).expect("b");
        let out = a.elementwise_binary(&b, BinaryOp::BitwiseAnd).expect("ok");
        assert_eq!(out.shape(), &[2]);
        assert_eq!(out.values(), &[0.0, 3.0]);
    }

    // ── array manipulation tests ────────────────────────────────────────

    #[test]
    fn reshape_basic() {
        let arr =
            UFuncArray::new(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).expect("arr");
        let out = arr.reshape(&[2, 3]).expect("reshape");
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.values(), arr.values());
    }

    #[test]
    fn reshape_infer_dimension() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");
        let out = arr.reshape(&[3, -1]).expect("reshape with -1");
        assert_eq!(out.shape(), &[3, 2]);
    }

    #[test]
    fn reshape_to_scalar() {
        let arr = UFuncArray::new(vec![1], vec![42.0], DType::F64).expect("arr");
        let out = arr.reshape(&[]).expect("reshape to scalar");
        assert_eq!(out.shape(), &[] as &[usize]);
        assert_eq!(out.values(), &[42.0]);
    }

    #[test]
    fn reshape_incompatible_count_fails() {
        let arr =
            UFuncArray::new(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).expect("arr");
        assert!(arr.reshape(&[2, 4]).is_err());
    }

    #[test]
    fn transpose_reverse_axes() {
        // np.array([[1,2,3],[4,5,6]]).T -> [[1,4],[2,5],[3,6]]
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");
        let out = arr.transpose(None).expect("transpose");
        assert_eq!(out.shape(), &[3, 2]);
        assert_eq!(out.values(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_specific_axes() {
        // 2x3x4 -> axes=[2,0,1] -> 4x2x3
        let arr = UFuncArray::new(
            vec![2, 3, 4],
            (0..24).map(|i| i as f64).collect(),
            DType::F64,
        )
        .expect("arr");
        let out = arr.transpose(Some(&[2, 0, 1])).expect("transpose axes");
        assert_eq!(out.shape(), &[4, 2, 3]);
        // Element [0,0,0] in new = element [0,0,0] in old = 0
        assert_eq!(out.values()[0], 0.0);
        // Element [1,0,0] in new = element [0,0,1] in old = 1
        assert_eq!(out.values()[6], 1.0);
    }

    #[test]
    fn transpose_identity() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");
        let out = arr.transpose(Some(&[0, 1])).expect("identity transpose");
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.values(), arr.values());
    }

    #[test]
    fn flatten_preserves_data() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");
        let out = arr.flatten();
        assert_eq!(out.shape(), &[6]);
        assert_eq!(out.values(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn squeeze_removes_all_ones() {
        let arr = UFuncArray::new(vec![1, 3, 1], vec![1.0, 2.0, 3.0], DType::F64).expect("arr");
        let out = arr.squeeze(None).expect("squeeze");
        assert_eq!(out.shape(), &[3]);
    }

    #[test]
    fn squeeze_specific_axis() {
        let arr = UFuncArray::new(vec![1, 3, 1], vec![1.0, 2.0, 3.0], DType::F64).expect("arr");
        let out = arr.squeeze(Some(0)).expect("squeeze axis=0");
        assert_eq!(out.shape(), &[3, 1]);
    }

    #[test]
    fn squeeze_rejects_non_one_axis() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");
        assert!(arr.squeeze(Some(1)).is_err());
    }

    #[test]
    fn expand_dims_front() {
        let arr = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).expect("arr");
        let out = arr.expand_dims(0).expect("expand front");
        assert_eq!(out.shape(), &[1, 3]);
    }

    #[test]
    fn expand_dims_back() {
        let arr = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).expect("arr");
        let out = arr.expand_dims(-1).expect("expand back");
        assert_eq!(out.shape(), &[3, 1]);
    }

    #[test]
    fn expand_dims_middle() {
        let arr = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("arr");
        let out = arr.expand_dims(1).expect("expand middle");
        assert_eq!(out.shape(), &[2, 1, 3]);
    }

    #[test]
    fn reshape_then_transpose_roundtrip() {
        let arr =
            UFuncArray::new(vec![6], (1..=6).map(|i| i as f64).collect(), DType::F64).expect("arr");
        let reshaped = arr.reshape(&[2, 3]).expect("reshape");
        let transposed = reshaped.transpose(None).expect("transpose");
        let back = transposed.transpose(None).expect("transpose back");
        assert_eq!(back.shape(), &[2, 3]);
        assert_eq!(back.values(), reshaped.values());
    }

    // ── where / sort / argsort / searchsorted / concatenate / stack tests

    #[test]
    fn where_select_basic() {
        // np.where([True, False, True], [1,2,3], [4,5,6]) = [1,5,3]
        let cond = UFuncArray::new(vec![3], vec![1.0, 0.0, 1.0], DType::Bool).expect("cond");
        let x = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).expect("x");
        let y = UFuncArray::new(vec![3], vec![4.0, 5.0, 6.0], DType::F64).expect("y");
        let out = UFuncArray::where_select(&cond, &x, &y).expect("where");
        assert_eq!(out.values(), &[1.0, 5.0, 3.0]);
    }

    #[test]
    fn where_select_broadcast_scalar() {
        // np.where([True, False], 10, 20) = [10, 20]
        let cond = UFuncArray::new(vec![2], vec![1.0, 0.0], DType::Bool).expect("cond");
        let x = UFuncArray::scalar(10.0, DType::F64);
        let y = UFuncArray::scalar(20.0, DType::F64);
        let out = UFuncArray::where_select(&cond, &x, &y).expect("where");
        assert_eq!(out.shape(), &[2]);
        assert_eq!(out.values(), &[10.0, 20.0]);
    }

    #[test]
    fn where_select_2d() {
        // np.where([[T,F],[F,T]], [[1,2],[3,4]], 0) = [[1,0],[0,4]]
        let cond =
            UFuncArray::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0], DType::Bool).expect("cond");
        let x = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).expect("x");
        let y = UFuncArray::scalar(0.0, DType::F64);
        let out = UFuncArray::where_select(&cond, &x, &y).expect("where");
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.values(), &[1.0, 0.0, 0.0, 4.0]);
    }

    #[test]
    fn sort_axis_none() {
        let arr = UFuncArray::new(vec![2, 3], vec![5.0, 1.0, 3.0, 2.0, 4.0, 0.0], DType::F64)
            .expect("arr");
        let out = arr.sort(None).expect("sort");
        assert_eq!(out.shape(), &[6]);
        assert_eq!(out.values(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn sort_axis_one() {
        // np.sort([[5,1,3],[2,4,0]], axis=1) = [[1,3,5],[0,2,4]]
        let arr = UFuncArray::new(vec![2, 3], vec![5.0, 1.0, 3.0, 2.0, 4.0, 0.0], DType::F64)
            .expect("arr");
        let out = arr.sort(Some(1)).expect("sort axis=1");
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.values(), &[1.0, 3.0, 5.0, 0.0, 2.0, 4.0]);
    }

    #[test]
    fn sort_axis_zero() {
        // np.sort([[5,1],[2,4]], axis=0) = [[2,1],[5,4]]
        let arr = UFuncArray::new(vec![2, 2], vec![5.0, 1.0, 2.0, 4.0], DType::F64).expect("arr");
        let out = arr.sort(Some(0)).expect("sort axis=0");
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.values(), &[2.0, 1.0, 5.0, 4.0]);
    }

    #[test]
    fn argsort_axis_none() {
        // np.argsort([3,1,2]) = [1,2,0]
        let arr = UFuncArray::new(vec![3], vec![3.0, 1.0, 2.0], DType::F64).expect("arr");
        let out = arr.argsort(None).expect("argsort");
        assert_eq!(out.values(), &[1.0, 2.0, 0.0]);
        assert_eq!(out.dtype(), DType::I64);
    }

    #[test]
    fn argsort_axis_one() {
        // np.argsort([[5,1,3],[2,4,0]], axis=1) = [[1,2,0],[2,0,1]]
        let arr = UFuncArray::new(vec![2, 3], vec![5.0, 1.0, 3.0, 2.0, 4.0, 0.0], DType::F64)
            .expect("arr");
        let out = arr.argsort(Some(1)).expect("argsort axis=1");
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.values(), &[1.0, 2.0, 0.0, 2.0, 0.0, 1.0]);
    }

    #[test]
    fn searchsorted_scalar_probe() {
        // np.searchsorted([1,3,5,7], 4) = 2
        let sorted =
            UFuncArray::new(vec![4], vec![1.0, 3.0, 5.0, 7.0], DType::F64).expect("sorted");
        let probe = UFuncArray::scalar(4.0, DType::F64);
        let out = sorted.searchsorted(&probe).expect("searchsorted");
        assert_eq!(out.shape(), &[]);
        assert_eq!(out.values(), &[2.0]);
        assert_eq!(out.dtype(), DType::I64);
    }

    #[test]
    fn searchsorted_vector_and_duplicates_left_side() {
        // np.searchsorted([1,2,2,2,3], [0,2,2.5,4]) = [0,1,4,5]
        let sorted =
            UFuncArray::new(vec![5], vec![1.0, 2.0, 2.0, 2.0, 3.0], DType::F64).expect("sorted");
        let probes =
            UFuncArray::new(vec![4], vec![0.0, 2.0, 2.5, 4.0], DType::F64).expect("probes");
        let out = sorted.searchsorted(&probes).expect("searchsorted");
        assert_eq!(out.shape(), &[4]);
        assert_eq!(out.values(), &[0.0, 1.0, 4.0, 5.0]);
        assert_eq!(out.dtype(), DType::I64);
    }

    #[test]
    fn searchsorted_preserves_probe_shape() {
        // np.searchsorted([1,3,5,7], [[0,4],[6,8]]) = [[0,2],[3,4]]
        let sorted =
            UFuncArray::new(vec![4], vec![1.0, 3.0, 5.0, 7.0], DType::F64).expect("sorted");
        let probes =
            UFuncArray::new(vec![2, 2], vec![0.0, 4.0, 6.0, 8.0], DType::F64).expect("probes");
        let out = sorted.searchsorted(&probes).expect("searchsorted");
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.values(), &[0.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn searchsorted_rejects_non_1d_sorted_input() {
        let not_1d =
            UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).expect("not_1d");
        let probe = UFuncArray::scalar(2.0, DType::F64);
        assert!(not_1d.searchsorted(&probe).is_err());
    }

    #[test]
    fn concatenate_axis_zero() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).expect("a");
        let b = UFuncArray::new(vec![1, 2], vec![5.0, 6.0], DType::F64).expect("b");
        let out = UFuncArray::concatenate(&[&a, &b], 0).expect("concat");
        assert_eq!(out.shape(), &[3, 2]);
        assert_eq!(out.values(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn concatenate_axis_one() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).expect("a");
        let b = UFuncArray::new(vec![2, 1], vec![5.0, 6.0], DType::F64).expect("b");
        let out = UFuncArray::concatenate(&[&a, &b], 1).expect("concat");
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.values(), &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn concatenate_three_arrays() {
        let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).expect("a");
        let b = UFuncArray::new(vec![3], vec![3.0, 4.0, 5.0], DType::F64).expect("b");
        let c = UFuncArray::new(vec![1], vec![6.0], DType::F64).expect("c");
        let out = UFuncArray::concatenate(&[&a, &b, &c], 0).expect("concat");
        assert_eq!(out.shape(), &[6]);
        assert_eq!(out.values(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn concatenate_rank_mismatch_fails() {
        let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).expect("a");
        let b = UFuncArray::new(vec![1, 2], vec![3.0, 4.0], DType::F64).expect("b");
        assert!(UFuncArray::concatenate(&[&a, &b], 0).is_err());
    }

    #[test]
    fn stack_axis_zero() {
        let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).expect("a");
        let b = UFuncArray::new(vec![2], vec![3.0, 4.0], DType::F64).expect("b");
        let out = UFuncArray::stack(&[&a, &b], 0).expect("stack axis=0");
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.values(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn stack_axis_one() {
        // np.stack([[1,2],[3,4]], axis=1) over 1D inputs -> [[1,3],[2,4]]
        let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).expect("a");
        let b = UFuncArray::new(vec![2], vec![3.0, 4.0], DType::F64).expect("b");
        let out = UFuncArray::stack(&[&a, &b], 1).expect("stack axis=1");
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.values(), &[1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn stack_negative_axis() {
        let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).expect("a");
        let b = UFuncArray::new(vec![2], vec![3.0, 4.0], DType::F64).expect("b");
        let out = UFuncArray::stack(&[&a, &b], -1).expect("stack axis=-1");
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.values(), &[1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn stack_promotes_dtype() {
        let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::I64).expect("a");
        let b = UFuncArray::new(vec![2], vec![0.5, 1.5], DType::F64).expect("b");
        let out = UFuncArray::stack(&[&a, &b], 0).expect("stack");
        assert_eq!(out.dtype(), DType::F64);
    }

    #[test]
    fn stack_shape_mismatch_fails() {
        let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).expect("a");
        let b = UFuncArray::new(vec![3], vec![3.0, 4.0, 5.0], DType::F64).expect("b");
        assert!(UFuncArray::stack(&[&a, &b], 0).is_err());
    }

    #[test]
    fn stack_empty_input_fails() {
        assert!(UFuncArray::stack(&[], 0).is_err());
    }

    #[test]
    fn packet005_log_record_is_replay_complete() {
        let record = UFuncLogRecord {
            fixture_id: "UP-005-elementwise-deterministic".to_string(),
            seed: 5001,
            mode: UFuncRuntimeMode::Strict,
            env_fingerprint: "fnp-ufunc-tests".to_string(),
            artifact_refs: packet005_artifacts(),
            reason_code: "ufunc_broadcast_selector_determinism".to_string(),
            passed: true,
        };
        assert!(record.is_replay_complete());
    }

    #[test]
    fn packet005_log_record_rejects_missing_required_fields() {
        let missing = UFuncLogRecord {
            fixture_id: String::new(),
            seed: 5002,
            mode: UFuncRuntimeMode::Hardened,
            env_fingerprint: String::new(),
            artifact_refs: Vec::new(),
            reason_code: String::new(),
            passed: false,
        };
        assert!(!missing.is_replay_complete());
    }

    #[test]
    fn packet005_reason_codes_round_trip_into_replay_logs() {
        for (idx, reason_code) in UFUNC_PACKET_REASON_CODES.iter().enumerate() {
            let record = UFuncLogRecord {
                fixture_id: format!("UP-005-{idx}"),
                seed: 5100 + u64::try_from(idx).expect("small index"),
                mode: UFuncRuntimeMode::Strict,
                env_fingerprint: "fnp-ufunc-tests".to_string(),
                artifact_refs: packet005_artifacts(),
                reason_code: (*reason_code).to_string(),
                passed: true,
            };
            assert!(record.is_replay_complete());
            assert_eq!(record.reason_code, *reason_code);
        }
    }

    // ── Array creation function tests ────────────────────────────────

    #[test]
    fn zeros_creates_correct_array() {
        let arr = UFuncArray::zeros(vec![2, 3], DType::F64).unwrap();
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.values(), &[0.0; 6]);
        assert_eq!(arr.dtype(), DType::F64);
    }

    #[test]
    fn zeros_like_matches_source() {
        let src =
            UFuncArray::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::I32).unwrap();
        let z = UFuncArray::zeros_like(&src);
        assert_eq!(z.shape(), &[3, 2]);
        assert_eq!(z.values(), &[0.0; 6]);
        assert_eq!(z.dtype(), DType::I32);
    }

    #[test]
    fn ones_creates_correct_array() {
        let arr = UFuncArray::ones(vec![4], DType::I64).unwrap();
        assert_eq!(arr.shape(), &[4]);
        assert_eq!(arr.values(), &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn ones_like_matches_source() {
        let src = UFuncArray::new(vec![2], vec![5.0, 6.0], DType::F32).unwrap();
        let o = UFuncArray::ones_like(&src);
        assert_eq!(o.shape(), &[2]);
        assert_eq!(o.values(), &[1.0, 1.0]);
        assert_eq!(o.dtype(), DType::F32);
    }

    #[test]
    fn full_fills_with_value() {
        let arr = UFuncArray::full(vec![2, 2], 42.0, DType::F64).unwrap();
        assert_eq!(arr.values(), &[42.0, 42.0, 42.0, 42.0]);
    }

    #[test]
    fn full_like_fills_with_value() {
        let src = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::I32).unwrap();
        let f = UFuncArray::full_like(&src, 7.0);
        assert_eq!(f.values(), &[7.0, 7.0, 7.0]);
        assert_eq!(f.dtype(), DType::I32);
    }

    #[test]
    fn empty_creates_zeroed_array() {
        let arr = UFuncArray::empty(vec![3], DType::F64).unwrap();
        assert_eq!(arr.shape(), &[3]);
        assert_eq!(arr.values().len(), 3);
    }

    #[test]
    fn arange_basic() {
        let arr = UFuncArray::arange(0.0, 5.0, 1.0, DType::F64).unwrap();
        assert_eq!(arr.shape(), &[5]);
        assert_eq!(arr.values(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn arange_with_step() {
        let arr = UFuncArray::arange(1.0, 10.0, 3.0, DType::I32).unwrap();
        assert_eq!(arr.shape(), &[3]);
        assert_eq!(arr.values(), &[1.0, 4.0, 7.0]);
    }

    #[test]
    fn arange_negative_step() {
        let arr = UFuncArray::arange(5.0, 0.0, -1.0, DType::F64).unwrap();
        assert_eq!(arr.shape(), &[5]);
        assert_eq!(arr.values(), &[5.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn arange_zero_step_errors() {
        let err = UFuncArray::arange(0.0, 5.0, 0.0, DType::F64).unwrap_err();
        assert!(err.to_string().contains("non-zero"));
    }

    #[test]
    fn arange_empty_when_start_equals_stop() {
        let arr = UFuncArray::arange(3.0, 3.0, 1.0, DType::F64).unwrap();
        assert_eq!(arr.shape(), &[0]);
        assert!(arr.values().is_empty());
    }

    #[test]
    fn linspace_basic() {
        let arr = UFuncArray::linspace(0.0, 1.0, 5, DType::F64).unwrap();
        assert_eq!(arr.shape(), &[5]);
        assert_eq!(arr.values(), &[0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn linspace_single_point() {
        let arr = UFuncArray::linspace(3.0, 7.0, 1, DType::F64).unwrap();
        assert_eq!(arr.shape(), &[1]);
        assert_eq!(arr.values(), &[3.0]);
    }

    #[test]
    fn linspace_empty() {
        let arr = UFuncArray::linspace(0.0, 1.0, 0, DType::F64).unwrap();
        assert_eq!(arr.shape(), &[0]);
        assert!(arr.values().is_empty());
    }

    #[test]
    fn eye_identity_3x3() {
        let arr = UFuncArray::eye(3, None, 0, DType::F64).unwrap();
        assert_eq!(arr.shape(), &[3, 3]);
        assert_eq!(arr.values(), &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn eye_rectangular() {
        let arr = UFuncArray::eye(2, Some(3), 0, DType::F64).unwrap();
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.values(), &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn eye_offset_diagonal() {
        let arr = UFuncArray::eye(3, None, 1, DType::F64).unwrap();
        assert_eq!(arr.shape(), &[3, 3]);
        assert_eq!(arr.values(), &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn eye_negative_diagonal() {
        let arr = UFuncArray::eye(3, None, -1, DType::F64).unwrap();
        assert_eq!(arr.shape(), &[3, 3]);
        assert_eq!(arr.values(), &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn diag_from_1d_creates_matrix() {
        let v = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let m = v.diag(0).unwrap();
        assert_eq!(m.shape(), &[3, 3]);
        assert_eq!(m.values(), &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn diag_from_2d_extracts_diagonal() {
        let m = UFuncArray::new(
            vec![3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            DType::F64,
        )
        .unwrap();
        let d = m.diag(0).unwrap();
        assert_eq!(d.shape(), &[3]);
        assert_eq!(d.values(), &[1.0, 5.0, 9.0]);
    }

    #[test]
    fn diag_from_2d_offset() {
        let m = UFuncArray::new(
            vec![3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            DType::F64,
        )
        .unwrap();
        let d = m.diag(1).unwrap();
        assert_eq!(d.shape(), &[2]);
        assert_eq!(d.values(), &[2.0, 6.0]);
    }

    #[test]
    fn triu_upper_triangle() {
        let m = UFuncArray::new(
            vec![3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            DType::F64,
        )
        .unwrap();
        let u = m.triu(0).unwrap();
        assert_eq!(u.values(), &[1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]);
    }

    #[test]
    fn tril_lower_triangle() {
        let m = UFuncArray::new(
            vec![3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            DType::F64,
        )
        .unwrap();
        let l = m.tril(0).unwrap();
        assert_eq!(l.values(), &[1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn zeros_scalar_shape() {
        let arr = UFuncArray::zeros(Vec::new(), DType::F64).unwrap();
        assert_eq!(arr.shape(), &[] as &[usize]);
        assert_eq!(arr.values(), &[0.0]);
    }

    // ── split/tile/repeat/roll/flip/unique tests ─────────────────────

    #[test]
    fn split_equal_sections() {
        let arr = UFuncArray::new(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let parts = arr.split(3, 0).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].values(), &[1.0, 2.0]);
        assert_eq!(parts[1].values(), &[3.0, 4.0]);
        assert_eq!(parts[2].values(), &[5.0, 6.0]);
    }

    #[test]
    fn split_2d_axis0() {
        let arr = UFuncArray::new(
            vec![4, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            DType::F64,
        )
        .unwrap();
        let parts = arr.split(2, 0).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].shape(), &[2, 2]);
        assert_eq!(parts[0].values(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(parts[1].values(), &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn split_uneven_fails() {
        let arr = UFuncArray::new(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0], DType::F64).unwrap();
        assert!(arr.split(3, 0).is_err());
    }

    #[test]
    fn tile_1d() {
        let arr = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let tiled = arr.tile(&[2]).unwrap();
        assert_eq!(tiled.shape(), &[6]);
        assert_eq!(tiled.values(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn tile_2d() {
        let arr = UFuncArray::new(vec![1, 2], vec![1.0, 2.0], DType::F64).unwrap();
        let tiled = arr.tile(&[2, 3]).unwrap();
        assert_eq!(tiled.shape(), &[2, 6]);
        assert_eq!(
            tiled.values(),
            &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        );
    }

    #[test]
    fn repeat_flat() {
        let arr = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let rep = arr.repeat(2, None).unwrap();
        assert_eq!(rep.shape(), &[6]);
        assert_eq!(rep.values(), &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn repeat_axis0() {
        let arr = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let rep = arr.repeat(2, Some(0)).unwrap();
        assert_eq!(rep.shape(), &[4, 2]);
        assert_eq!(rep.values(), &[1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn repeat_axis1() {
        let arr = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let rep = arr.repeat(3, Some(1)).unwrap();
        assert_eq!(rep.shape(), &[2, 6]);
        assert_eq!(
            rep.values(),
            &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0]
        );
    }

    #[test]
    fn roll_flat() {
        let arr = UFuncArray::new(vec![5], vec![0.0, 1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let rolled = arr.roll(2, None).unwrap();
        assert_eq!(rolled.values(), &[3.0, 4.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn roll_negative() {
        let arr = UFuncArray::new(vec![5], vec![0.0, 1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let rolled = arr.roll(-1, None).unwrap();
        assert_eq!(rolled.values(), &[1.0, 2.0, 3.0, 4.0, 0.0]);
    }

    #[test]
    fn roll_axis0() {
        let arr =
            UFuncArray::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let rolled = arr.roll(1, Some(0)).unwrap();
        assert_eq!(rolled.values(), &[5.0, 6.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn flip_flat() {
        let arr = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let flipped = arr.flip(None).unwrap();
        assert_eq!(flipped.values(), &[4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn flip_axis0_2d() {
        let arr =
            UFuncArray::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let flipped = arr.flip(Some(0)).unwrap();
        assert_eq!(flipped.values(), &[5.0, 6.0, 3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn flip_axis1_2d() {
        let arr =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let flipped = arr.flip(Some(1)).unwrap();
        assert_eq!(flipped.values(), &[3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
    }

    #[test]
    fn unique_removes_duplicates() {
        let arr = UFuncArray::new(vec![6], vec![3.0, 1.0, 2.0, 1.0, 3.0, 2.0], DType::F64).unwrap();
        let u = arr.unique();
        assert_eq!(u.shape(), &[3]);
        assert_eq!(u.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn unique_already_unique() {
        let arr = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let u = arr.unique();
        assert_eq!(u.values(), &[1.0, 2.0, 3.0]);
    }

    // ── dot / matmul / outer / inner / trace ─────────────────────────

    #[test]
    fn dot_1d_1d() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![4.0, 5.0, 6.0], DType::F64).unwrap();
        let r = a.dot(&b).unwrap();
        assert!(r.shape().is_empty()); // 0-d scalar
        assert_eq!(r.values(), &[32.0]); // 1*4 + 2*5 + 3*6
    }

    #[test]
    fn dot_1d_1d_length_mismatch() {
        let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![4.0, 5.0, 6.0], DType::F64).unwrap();
        assert!(a.dot(&b).is_err());
    }

    #[test]
    fn dot_2d_2d() {
        // [[1,2],[3,4]] . [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0], DType::F64).unwrap();
        let r = a.dot(&b).unwrap();
        assert_eq!(r.shape(), &[2, 2]);
        assert_eq!(r.values(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn dot_1d_2d() {
        // [1,2] . [[3,4],[5,6]] = [13, 16]
        let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2, 2], vec![3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let r = a.dot(&b).unwrap();
        assert_eq!(r.shape(), &[2]);
        assert_eq!(r.values(), &[13.0, 16.0]);
    }

    #[test]
    fn dot_2d_1d() {
        // [[1,2],[3,4]] . [5,6] = [17, 39]
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2], vec![5.0, 6.0], DType::F64).unwrap();
        let r = a.dot(&b).unwrap();
        assert_eq!(r.shape(), &[2]);
        assert_eq!(r.values(), &[17.0, 39.0]);
    }

    #[test]
    fn matmul_2x3_by_3x2() {
        // [[1,2,3],[4,5,6]] @ [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let b = UFuncArray::new(
            vec![3, 2],
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            DType::F64,
        )
        .unwrap();
        let r = a.matmul(&b).unwrap();
        assert_eq!(r.shape(), &[2, 2]);
        assert_eq!(r.values(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_dimension_mismatch() {
        let a = UFuncArray::new(vec![2, 3], vec![1.0; 6], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2, 2], vec![1.0; 4], DType::F64).unwrap();
        assert!(a.matmul(&b).is_err());
    }

    #[test]
    fn matmul_identity() {
        // A @ I = A
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let eye = UFuncArray::eye(2, None, 0, DType::F64).unwrap();
        let r = a.matmul(&eye).unwrap();
        assert_eq!(r.values(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn outer_product() {
        // outer([1,2,3], [4,5]) = [[4,5],[8,10],[12,15]]
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2], vec![4.0, 5.0], DType::F64).unwrap();
        let r = a.outer(&b).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(r.values(), &[4.0, 5.0, 8.0, 10.0, 12.0, 15.0]);
    }

    #[test]
    fn outer_requires_1d() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0; 4], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap();
        assert!(a.outer(&b).is_err());
    }

    #[test]
    fn inner_1d() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![4.0, 5.0, 6.0], DType::F64).unwrap();
        let r = a.inner(&b).unwrap();
        assert_eq!(r.values(), &[32.0]);
    }

    #[test]
    fn inner_rejects_2d() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0; 4], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2, 2], vec![1.0; 4], DType::F64).unwrap();
        assert!(a.inner(&b).is_err());
    }

    #[test]
    fn trace_identity() {
        let eye = UFuncArray::eye(3, None, 0, DType::F64).unwrap();
        let t = eye.trace(0).unwrap();
        assert_eq!(t.values(), &[3.0]);
    }

    #[test]
    fn trace_offset() {
        // [[1,2,3],[4,5,6],[7,8,9]]
        let a = UFuncArray::new(
            vec![3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            DType::F64,
        )
        .unwrap();
        assert_eq!(a.trace(0).unwrap().values(), &[15.0]); // 1+5+9
        assert_eq!(a.trace(1).unwrap().values(), &[8.0]); // 2+6
        assert_eq!(a.trace(-1).unwrap().values(), &[12.0]); // 4+8
    }

    #[test]
    fn dot_dtype_promotion() {
        let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::I32).unwrap();
        let b = UFuncArray::new(vec![2], vec![3.0, 4.0], DType::F32).unwrap();
        let r = a.dot(&b).unwrap();
        assert_eq!(r.dtype(), DType::F64);
    }

    // ── fancy / boolean indexing ─────────────────────────────────────

    #[test]
    fn take_flat() {
        let a = UFuncArray::new(vec![5], vec![10.0, 20.0, 30.0, 40.0, 50.0], DType::F64).unwrap();
        let r = a.take(&[0, 2, 4], None).unwrap();
        assert_eq!(r.shape(), &[3]);
        assert_eq!(r.values(), &[10.0, 30.0, 50.0]);
    }

    #[test]
    fn take_flat_negative() {
        let a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let r = a.take(&[-1, -2], None).unwrap();
        assert_eq!(r.values(), &[4.0, 3.0]);
    }

    #[test]
    fn take_flat_out_of_bounds() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        assert!(a.take(&[5], None).is_err());
    }

    #[test]
    fn take_axis0_2d() {
        // [[1,2],[3,4],[5,6]], take rows [2,0]
        let a =
            UFuncArray::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let r = a.take(&[2, 0], Some(0)).unwrap();
        assert_eq!(r.shape(), &[2, 2]);
        assert_eq!(r.values(), &[5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn take_axis1_2d() {
        // [[1,2,3],[4,5,6]], take cols [2, 0]
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let r = a.take(&[2, 0], Some(1)).unwrap();
        assert_eq!(r.shape(), &[2, 2]);
        assert_eq!(r.values(), &[3.0, 1.0, 6.0, 4.0]);
    }

    #[test]
    fn take_duplicate_indices() {
        let a = UFuncArray::new(vec![3], vec![10.0, 20.0, 30.0], DType::F64).unwrap();
        let r = a.take(&[1, 1, 1], None).unwrap();
        assert_eq!(r.values(), &[20.0, 20.0, 20.0]);
    }

    #[test]
    fn compress_flat() {
        let a = UFuncArray::new(vec![5], vec![10.0, 20.0, 30.0, 40.0, 50.0], DType::F64).unwrap();
        let r = a.compress(&[true, false, true, false, true], None).unwrap();
        assert_eq!(r.values(), &[10.0, 30.0, 50.0]);
    }

    #[test]
    fn compress_axis0() {
        // [[1,2],[3,4],[5,6]] compress [true, false, true] on axis 0
        let a =
            UFuncArray::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let r = a.compress(&[true, false, true], Some(0)).unwrap();
        assert_eq!(r.shape(), &[2, 2]);
        assert_eq!(r.values(), &[1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn compress_length_mismatch() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        assert!(a.compress(&[true, false], None).is_err());
    }

    #[test]
    fn boolean_index_mask() {
        let a = UFuncArray::new(vec![4], vec![10.0, 20.0, 30.0, 40.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![4], vec![1.0, 0.0, 1.0, 0.0], DType::Bool).unwrap();
        let r = a.boolean_index(&mask).unwrap();
        assert_eq!(r.values(), &[10.0, 30.0]);
    }

    #[test]
    fn boolean_index_2d_flat() {
        // 2D array boolean indexed flattens
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let mask =
            UFuncArray::new(vec![2, 3], vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0], DType::Bool).unwrap();
        let r = a.boolean_index(&mask).unwrap();
        assert_eq!(r.values(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn boolean_set_scalar() {
        let mut a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![4], vec![0.0, 1.0, 0.0, 1.0], DType::Bool).unwrap();
        a.boolean_set(&mask, 99.0).unwrap();
        assert_eq!(a.values(), &[1.0, 99.0, 3.0, 99.0]);
    }

    #[test]
    fn fancy_index_flat() {
        let a = UFuncArray::new(vec![5], vec![10.0, 20.0, 30.0, 40.0, 50.0], DType::F64).unwrap();
        let r = a.fancy_index(&[4, 3, 2, 1, 0]).unwrap();
        assert_eq!(r.values(), &[50.0, 40.0, 30.0, 20.0, 10.0]);
    }

    #[test]
    fn fancy_set_flat() {
        let mut a = UFuncArray::new(vec![4], vec![0.0; 4], DType::F64).unwrap();
        a.fancy_set(&[0, 2], &[10.0, 30.0]).unwrap();
        assert_eq!(a.values(), &[10.0, 0.0, 30.0, 0.0]);
    }

    #[test]
    fn fancy_set_negative_index() {
        let mut a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        a.fancy_set(&[-1], &[99.0]).unwrap();
        assert_eq!(a.values(), &[1.0, 2.0, 99.0]);
    }

    #[test]
    fn fancy_set_out_of_bounds() {
        let mut a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        assert!(a.fancy_set(&[5], &[0.0]).is_err());
    }

    #[test]
    fn take_empty_indices() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let r = a.take(&[], None).unwrap();
        assert_eq!(r.shape(), &[0]);
        assert!(r.values().is_empty());
    }

    #[test]
    fn boolean_index_all_false() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![3], vec![0.0, 0.0, 0.0], DType::Bool).unwrap();
        let r = a.boolean_index(&mask).unwrap();
        assert_eq!(r.shape(), &[0]);
        assert!(r.values().is_empty());
    }

    // ── astype / any / all / nonzero / diff / isclose / allclose ─────

    #[test]
    fn astype_f64_to_i32() {
        let a = UFuncArray::new(vec![3], vec![1.7, -2.3, 3.9], DType::F64).unwrap();
        let r = a.astype(DType::I32);
        assert_eq!(r.dtype(), DType::I32);
        assert_eq!(r.values(), &[1.0, -2.0, 3.0]); // truncated
    }

    #[test]
    fn astype_to_bool() {
        let a = UFuncArray::new(vec![4], vec![0.0, 1.5, -1.0, 0.0], DType::F64).unwrap();
        let r = a.astype(DType::Bool);
        assert_eq!(r.values(), &[0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn astype_preserves_shape() {
        let a = UFuncArray::new(vec![2, 3], vec![1.0; 6], DType::F64).unwrap();
        let r = a.astype(DType::F32);
        assert_eq!(r.shape(), &[2, 3]);
    }

    #[test]
    fn any_all_true() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let r = a.any(None).unwrap();
        assert_eq!(r.values(), &[1.0]);
    }

    #[test]
    fn any_with_zeros() {
        let a = UFuncArray::new(vec![3], vec![0.0, 0.0, 1.0], DType::F64).unwrap();
        assert_eq!(a.any(None).unwrap().values(), &[1.0]);
    }

    #[test]
    fn any_all_false() {
        let a = UFuncArray::new(vec![3], vec![0.0, 0.0, 0.0], DType::F64).unwrap();
        assert_eq!(a.any(None).unwrap().values(), &[0.0]);
    }

    #[test]
    fn all_true() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        assert_eq!(a.all(None).unwrap().values(), &[1.0]);
    }

    #[test]
    fn all_with_zero() {
        let a = UFuncArray::new(vec![3], vec![1.0, 0.0, 3.0], DType::F64).unwrap();
        assert_eq!(a.all(None).unwrap().values(), &[0.0]);
    }

    #[test]
    fn any_axis0() {
        // [[0, 1], [0, 0]] → any(axis=0) = [0, 1]
        let a = UFuncArray::new(vec![2, 2], vec![0.0, 1.0, 0.0, 0.0], DType::F64).unwrap();
        let r = a.any(Some(0)).unwrap();
        assert_eq!(r.shape(), &[2]);
        assert_eq!(r.values(), &[0.0, 1.0]);
    }

    #[test]
    fn all_axis1() {
        // [[1, 1], [1, 0]] → all(axis=1) = [1, 0]
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 1.0, 1.0, 0.0], DType::F64).unwrap();
        let r = a.all(Some(1)).unwrap();
        assert_eq!(r.shape(), &[2]);
        assert_eq!(r.values(), &[1.0, 0.0]);
    }

    #[test]
    fn nonzero_basic() {
        let a = UFuncArray::new(vec![5], vec![0.0, 3.0, 0.0, 7.0, 0.0], DType::F64).unwrap();
        let r = a.nonzero();
        assert_eq!(r.values(), &[1.0, 3.0]); // indices of non-zero
        assert_eq!(r.dtype(), DType::I64);
    }

    #[test]
    fn nonzero_all_zero() {
        let a = UFuncArray::new(vec![3], vec![0.0, 0.0, 0.0], DType::F64).unwrap();
        let r = a.nonzero();
        assert!(r.values().is_empty());
    }

    #[test]
    fn diff_1d() {
        let a = UFuncArray::new(vec![5], vec![1.0, 3.0, 6.0, 10.0, 15.0], DType::F64).unwrap();
        let r = a.diff(1, None).unwrap();
        assert_eq!(r.shape(), &[4]);
        assert_eq!(r.values(), &[2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn diff_n2() {
        let a = UFuncArray::new(vec![5], vec![1.0, 3.0, 6.0, 10.0, 15.0], DType::F64).unwrap();
        let r = a.diff(2, None).unwrap();
        assert_eq!(r.shape(), &[3]);
        assert_eq!(r.values(), &[1.0, 1.0, 1.0]); // second differences
    }

    #[test]
    fn diff_2d_axis0() {
        // [[1,2],[4,5],[9,10]] → diff axis 0 → [[3,3],[5,5]]
        let a =
            UFuncArray::new(vec![3, 2], vec![1.0, 2.0, 4.0, 5.0, 9.0, 10.0], DType::F64).unwrap();
        let r = a.diff(1, Some(0)).unwrap();
        assert_eq!(r.shape(), &[2, 2]);
        assert_eq!(r.values(), &[3.0, 3.0, 5.0, 5.0]);
    }

    #[test]
    fn diff_2d_axis1() {
        // [[1,3,6],[2,5,9]] → diff axis 1 → [[2,3],[3,4]]
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 3.0, 6.0, 2.0, 5.0, 9.0], DType::F64).unwrap();
        let r = a.diff(1, Some(1)).unwrap();
        assert_eq!(r.shape(), &[2, 2]);
        assert_eq!(r.values(), &[2.0, 3.0, 3.0, 4.0]);
    }

    #[test]
    fn diff_n0_returns_copy() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let r = a.diff(0, None).unwrap();
        assert_eq!(r.values(), a.values());
    }

    #[test]
    fn isclose_basic() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![1.0, 2.0001, 3.1], DType::F64).unwrap();
        let r = a.isclose(&b, 1e-3, 1e-8).unwrap();
        assert_eq!(r.values(), &[1.0, 1.0, 0.0]); // 3.1 is not close to 3.0
    }

    #[test]
    fn allclose_true() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        assert!(a.allclose(&b, 1e-5, 1e-8).unwrap());
    }

    #[test]
    fn allclose_false() {
        let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2], vec![1.0, 3.0], DType::F64).unwrap();
        assert!(!a.allclose(&b, 1e-5, 1e-8).unwrap());
    }

    #[test]
    fn isclose_size_mismatch() {
        let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        assert!(a.isclose(&b, 1e-5, 1e-8).is_err());
    }

    // ── median / percentile / cummin / cummax / pad ─────────────────

    #[test]
    fn median_odd() {
        let a = UFuncArray::new(vec![5], vec![3.0, 1.0, 4.0, 1.0, 5.0], DType::F64).unwrap();
        let r = a.median(None).unwrap();
        assert_eq!(r.values(), &[3.0]);
    }

    #[test]
    fn median_even() {
        let a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let r = a.median(None).unwrap();
        assert_eq!(r.values(), &[2.5]);
    }

    #[test]
    fn median_axis0() {
        // [[1,4],[2,5],[3,6]] → median(axis=0) = [2, 5]
        let a =
            UFuncArray::new(vec![3, 2], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], DType::F64).unwrap();
        let r = a.median(Some(0)).unwrap();
        assert_eq!(r.shape(), &[2]);
        assert_eq!(r.values(), &[2.0, 5.0]);
    }

    #[test]
    fn percentile_0_and_100() {
        let a = UFuncArray::new(vec![4], vec![10.0, 20.0, 30.0, 40.0], DType::F64).unwrap();
        assert_eq!(a.percentile(0.0, None).unwrap().values(), &[10.0]);
        assert_eq!(a.percentile(100.0, None).unwrap().values(), &[40.0]);
    }

    #[test]
    fn percentile_50_is_median() {
        let a = UFuncArray::new(vec![5], vec![3.0, 1.0, 4.0, 1.0, 5.0], DType::F64).unwrap();
        let med = a.median(None).unwrap();
        let p50 = a.percentile(50.0, None).unwrap();
        assert_eq!(med.values(), p50.values());
    }

    #[test]
    fn percentile_25() {
        // [1,2,3,4] → p25 = 1.75 (linear interpolation)
        let a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let r = a.percentile(25.0, None).unwrap();
        assert!((r.values()[0] - 1.75).abs() < 1e-10);
    }

    #[test]
    fn percentile_out_of_range() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        assert!(a.percentile(101.0, None).is_err());
        assert!(a.percentile(-1.0, None).is_err());
    }

    #[test]
    fn cummin_flat() {
        let a = UFuncArray::new(vec![5], vec![3.0, 1.0, 4.0, 1.0, 5.0], DType::F64).unwrap();
        let r = a.cummin(None).unwrap();
        assert_eq!(r.values(), &[3.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn cummax_flat() {
        let a = UFuncArray::new(vec![5], vec![1.0, 3.0, 2.0, 5.0, 4.0], DType::F64).unwrap();
        let r = a.cummax(None).unwrap();
        assert_eq!(r.values(), &[1.0, 3.0, 3.0, 5.0, 5.0]);
    }

    #[test]
    fn cummin_axis0() {
        // [[3,4],[1,5],[2,2]] cummin axis 0 → [[3,4],[1,4],[1,2]]
        let a =
            UFuncArray::new(vec![3, 2], vec![3.0, 4.0, 1.0, 5.0, 2.0, 2.0], DType::F64).unwrap();
        let r = a.cummin(Some(0)).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(r.values(), &[3.0, 4.0, 1.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn cummax_axis1() {
        // [[1,3,2],[4,2,5]] cummax axis 1 → [[1,3,3],[4,4,5]]
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 3.0, 2.0, 4.0, 2.0, 5.0], DType::F64).unwrap();
        let r = a.cummax(Some(1)).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.values(), &[1.0, 3.0, 3.0, 4.0, 4.0, 5.0]);
    }

    #[test]
    fn pad_1d() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let r = a.pad(&[(2, 1)], 0.0).unwrap();
        assert_eq!(r.shape(), &[6]);
        assert_eq!(r.values(), &[0.0, 0.0, 1.0, 2.0, 3.0, 0.0]);
    }

    #[test]
    fn pad_2d() {
        // [[1,2],[3,4]] pad (1,1) on each axis with -1
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let r = a.pad(&[(1, 1), (1, 1)], -1.0).unwrap();
        assert_eq!(r.shape(), &[4, 4]);
        #[rustfmt::skip]
        let expected = vec![
            -1.0, -1.0, -1.0, -1.0,
            -1.0,  1.0,  2.0, -1.0,
            -1.0,  3.0,  4.0, -1.0,
            -1.0, -1.0, -1.0, -1.0,
        ];
        assert_eq!(r.values(), &expected);
    }

    #[test]
    fn pad_wrong_ndim() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        assert!(a.pad(&[(1, 1), (1, 1)], 0.0).is_err()); // 2 pad widths for 1-D
    }

    // ── meshgrid / gradient / histogram / bincount / interp ─────────

    #[test]
    fn meshgrid_2d() {
        let x = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let y = UFuncArray::new(vec![2], vec![4.0, 5.0], DType::F64).unwrap();
        let grids = UFuncArray::meshgrid(&[x, y]).unwrap();
        assert_eq!(grids.len(), 2);
        // xx shape should be (len(y), len(x)) = (2, 3)
        assert_eq!(grids[0].shape(), &[2, 3]);
        assert_eq!(grids[0].values(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        // yy shape should be (2, 3)
        assert_eq!(grids[1].shape(), &[2, 3]);
        assert_eq!(grids[1].values(), &[4.0, 4.0, 4.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn meshgrid_rejects_non_1d() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0; 4], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![1.0; 3], DType::F64).unwrap();
        assert!(UFuncArray::meshgrid(&[a, b]).is_err());
    }

    #[test]
    fn gradient_linear() {
        // gradient of [0, 1, 2, 3, 4] = [1, 1, 1, 1, 1]
        let a = UFuncArray::new(vec![5], vec![0.0, 1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let r = a.gradient().unwrap();
        assert_eq!(r.values(), &[1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn gradient_quadratic() {
        // gradient of [0, 1, 4, 9] = [1, 2, 4, 5]
        let a = UFuncArray::new(vec![4], vec![0.0, 1.0, 4.0, 9.0], DType::F64).unwrap();
        let r = a.gradient().unwrap();
        assert_eq!(r.values(), &[1.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn gradient_rejects_short() {
        let a = UFuncArray::new(vec![1], vec![1.0], DType::F64).unwrap();
        assert!(a.gradient().is_err());
    }

    #[test]
    fn histogram_uniform() {
        let a = UFuncArray::new(vec![4], vec![0.5, 1.5, 2.5, 3.5], DType::F64).unwrap();
        let (counts, edges) = a.histogram(4).unwrap();
        assert_eq!(counts.shape(), &[4]);
        // Each bin should have 1 element
        assert_eq!(counts.values(), &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(edges.shape(), &[5]);
    }

    #[test]
    fn histogram_single_value() {
        let a = UFuncArray::new(vec![3], vec![5.0, 5.0, 5.0], DType::F64).unwrap();
        let (counts, _) = a.histogram(3).unwrap();
        // All values in same bin
        let total: f64 = counts.values().iter().sum();
        assert_eq!(total, 3.0);
    }

    #[test]
    fn bincount_basic() {
        let a =
            UFuncArray::new(vec![7], vec![0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0], DType::I64).unwrap();
        let r = a.bincount().unwrap();
        assert_eq!(r.values(), &[1.0, 2.0, 3.0, 1.0]);
    }

    #[test]
    fn bincount_rejects_negative() {
        let a = UFuncArray::new(vec![2], vec![-1.0, 0.0], DType::I64).unwrap();
        assert!(a.bincount().is_err());
    }

    #[test]
    fn interp_basic() {
        let x = UFuncArray::new(vec![3], vec![0.5, 1.5, 2.5], DType::F64).unwrap();
        let xp = UFuncArray::new(vec![3], vec![0.0, 1.0, 2.0], DType::F64).unwrap();
        let fp = UFuncArray::new(vec![3], vec![0.0, 10.0, 20.0], DType::F64).unwrap();
        let r = UFuncArray::interp(&x, &xp, &fp).unwrap();
        assert_eq!(r.values(), &[5.0, 15.0, 20.0]); // 2.5 clamps to 20.0
    }

    #[test]
    fn interp_clamp_edges() {
        let x = UFuncArray::new(vec![2], vec![-1.0, 5.0], DType::F64).unwrap();
        let xp = UFuncArray::new(vec![2], vec![0.0, 1.0], DType::F64).unwrap();
        let fp = UFuncArray::new(vec![2], vec![10.0, 20.0], DType::F64).unwrap();
        let r = UFuncArray::interp(&x, &xp, &fp).unwrap();
        assert_eq!(r.values(), &[10.0, 20.0]); // clamped to edges
    }

    #[test]
    fn interp_exact_points() {
        let x = UFuncArray::new(vec![3], vec![0.0, 1.0, 2.0], DType::F64).unwrap();
        let xp = UFuncArray::new(vec![3], vec![0.0, 1.0, 2.0], DType::F64).unwrap();
        let fp = UFuncArray::new(vec![3], vec![100.0, 200.0, 300.0], DType::F64).unwrap();
        let r = UFuncArray::interp(&x, &xp, &fp).unwrap();
        assert_eq!(r.values(), &[100.0, 200.0, 300.0]);
    }

    // ── convolve / correlate / polyval / cross / vstack / hstack ─────

    #[test]
    fn convolve_basic() {
        // [1, 2, 3] * [0, 1, 0.5] = [0, 1, 2.5, 4, 1.5]
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let k = UFuncArray::new(vec![3], vec![0.0, 1.0, 0.5], DType::F64).unwrap();
        let r = a.convolve(&k).unwrap();
        assert_eq!(r.shape(), &[5]);
        assert_eq!(r.values(), &[0.0, 1.0, 2.5, 4.0, 1.5]);
    }

    #[test]
    fn convolve_identity() {
        // Convolving with [1] returns the same signal
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let k = UFuncArray::new(vec![1], vec![1.0], DType::F64).unwrap();
        let r = a.convolve(&k).unwrap();
        assert_eq!(r.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn correlate_basic() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![0.0, 1.0, 0.5], DType::F64).unwrap();
        let r = a.correlate(&b).unwrap();
        assert_eq!(r.shape(), &[5]);
        // correlate reverses kernel: convolve([1,2,3], [0.5,1,0])
        assert_eq!(r.values(), &[0.5, 2.0, 3.5, 3.0, 0.0]);
    }

    #[test]
    fn polyval_quadratic() {
        // p(x) = x^2 + 2x + 1 → coeffs = [1, 2, 1]
        let c = UFuncArray::new(vec![3], vec![1.0, 2.0, 1.0], DType::F64).unwrap();
        let x = UFuncArray::new(vec![3], vec![0.0, 1.0, 2.0], DType::F64).unwrap();
        let r = UFuncArray::polyval(&c, &x).unwrap();
        assert_eq!(r.values(), &[1.0, 4.0, 9.0]); // (x+1)^2
    }

    #[test]
    fn polyval_constant() {
        let c = UFuncArray::new(vec![1], vec![42.0], DType::F64).unwrap();
        let x = UFuncArray::new(vec![3], vec![0.0, 1.0, 100.0], DType::F64).unwrap();
        let r = UFuncArray::polyval(&c, &x).unwrap();
        assert_eq!(r.values(), &[42.0, 42.0, 42.0]);
    }

    #[test]
    fn cross_product() {
        // [1,0,0] × [0,1,0] = [0,0,1]
        let a = UFuncArray::new(vec![3], vec![1.0, 0.0, 0.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![0.0, 1.0, 0.0], DType::F64).unwrap();
        let r = a.cross(&b).unwrap();
        assert_eq!(r.values(), &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn cross_anticommutative() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![4.0, 5.0, 6.0], DType::F64).unwrap();
        let ab = a.cross(&b).unwrap();
        let ba = b.cross(&a).unwrap();
        // a × b = -(b × a)
        for (x, y) in ab.values().iter().zip(ba.values()) {
            assert!((x + y).abs() < 1e-10);
        }
    }

    #[test]
    fn cross_wrong_length() {
        let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        assert!(a.cross(&b).is_err());
    }

    #[test]
    fn vstack_1d() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![4.0, 5.0, 6.0], DType::F64).unwrap();
        let r = UFuncArray::vstack(&[a, b]).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.values(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn vstack_2d() {
        let a = UFuncArray::new(vec![1, 3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let b =
            UFuncArray::new(vec![2, 3], vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0], DType::F64).unwrap();
        let r = UFuncArray::vstack(&[a, b]).unwrap();
        assert_eq!(r.shape(), &[3, 3]);
    }

    #[test]
    fn hstack_1d() {
        let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![3.0, 4.0, 5.0], DType::F64).unwrap();
        let r = UFuncArray::hstack(&[a, b]).unwrap();
        assert_eq!(r.shape(), &[5]);
        assert_eq!(r.values(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn hstack_2d() {
        let a = UFuncArray::new(vec![2, 1], vec![1.0, 2.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2, 2], vec![3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let r = UFuncArray::hstack(&[a, b]).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.values(), &[1.0, 3.0, 4.0, 2.0, 5.0, 6.0]);
    }

    // ── slice / item / ravel / fill / ptp / round / choose ──────────

    #[test]
    fn slice_axis_basic() {
        let a = UFuncArray::new(vec![5], vec![10.0, 20.0, 30.0, 40.0, 50.0], DType::F64).unwrap();
        let r = a.slice_axis(0, Some(1), Some(4), 1).unwrap();
        assert_eq!(r.values(), &[20.0, 30.0, 40.0]);
    }

    #[test]
    fn slice_axis_step() {
        let a = UFuncArray::new(vec![6], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], DType::F64).unwrap();
        let r = a.slice_axis(0, Some(0), Some(6), 2).unwrap();
        assert_eq!(r.values(), &[0.0, 2.0, 4.0]);
    }

    #[test]
    fn slice_axis_negative() {
        let a = UFuncArray::new(vec![5], vec![10.0, 20.0, 30.0, 40.0, 50.0], DType::F64).unwrap();
        let r = a.slice_axis(0, Some(-3), None, 1).unwrap();
        assert_eq!(r.values(), &[30.0, 40.0, 50.0]);
    }

    #[test]
    fn slice_2d_rows() {
        // [[1,2],[3,4],[5,6]], take rows 0..2
        let a =
            UFuncArray::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let r = a.slice_axis(0, Some(0), Some(2), 1).unwrap();
        assert_eq!(r.shape(), &[2, 2]);
        assert_eq!(r.values(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn item_2d() {
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        assert_eq!(a.item(&[0, 2]).unwrap(), 3.0);
        assert_eq!(a.item(&[1, 0]).unwrap(), 4.0);
        assert_eq!(a.item(&[-1, -1]).unwrap(), 6.0);
    }

    #[test]
    fn item_out_of_bounds() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        assert!(a.item(&[5]).is_err());
    }

    #[test]
    fn itemset_2d() {
        let mut a = UFuncArray::new(vec![2, 2], vec![0.0; 4], DType::F64).unwrap();
        a.itemset(&[0, 1], 42.0).unwrap();
        a.itemset(&[1, 0], 99.0).unwrap();
        assert_eq!(a.values(), &[0.0, 42.0, 99.0, 0.0]);
    }

    #[test]
    fn ravel_returns_flat() {
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let r = a.ravel();
        assert_eq!(r.shape(), &[6]);
        assert_eq!(r.values(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn fill_array() {
        let mut a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        a.fill(7.0);
        assert_eq!(a.values(), &[7.0, 7.0, 7.0]);
    }

    #[test]
    fn ptp_flat() {
        let a = UFuncArray::new(vec![5], vec![3.0, 1.0, 4.0, 1.0, 5.0], DType::F64).unwrap();
        let r = a.ptp(None).unwrap();
        assert_eq!(r.values(), &[4.0]); // 5 - 1
    }

    #[test]
    fn ptp_axis0() {
        // [[1, 5], [3, 2]] → ptp(axis=0) = [2, 3]
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 5.0, 3.0, 2.0], DType::F64).unwrap();
        let r = a.ptp(Some(0)).unwrap();
        assert_eq!(r.shape(), &[2]);
        assert_eq!(r.values(), &[2.0, 3.0]);
    }

    #[test]
    fn round_to_decimals() {
        let a = UFuncArray::new(vec![3], vec![1.234, 5.678, 9.999], DType::F64).unwrap();
        let r = a.round_to(2);
        assert_eq!(r.values(), &[1.23, 5.68, 10.0]);
    }

    #[test]
    fn round_to_negative_decimals() {
        let a = UFuncArray::new(vec![3], vec![123.0, 456.0, 789.0], DType::F64).unwrap();
        let r = a.round_to(-2);
        assert_eq!(r.values(), &[100.0, 500.0, 800.0]);
    }

    #[test]
    fn choose_basic() {
        // index = [0, 1, 0, 1], choices = [[10,20,30,40], [50,60,70,80]]
        let idx = UFuncArray::new(vec![4], vec![0.0, 1.0, 0.0, 1.0], DType::I64).unwrap();
        let c0 = UFuncArray::new(vec![4], vec![10.0, 20.0, 30.0, 40.0], DType::F64).unwrap();
        let c1 = UFuncArray::new(vec![4], vec![50.0, 60.0, 70.0, 80.0], DType::F64).unwrap();
        let r = idx.choose(&[c0, c1]).unwrap();
        assert_eq!(r.values(), &[10.0, 60.0, 30.0, 80.0]);
    }

    #[test]
    fn choose_out_of_range() {
        let idx = UFuncArray::new(vec![2], vec![0.0, 5.0], DType::I64).unwrap();
        let c0 = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap();
        assert!(idx.choose(&[c0]).is_err());
    }

    // ── array creation: logspace, geomspace, fromfunction, indices, identity, tri, diagflat ──

    #[test]
    fn logspace_basic() {
        let r = UFuncArray::logspace(0.0, 2.0, 3, 10.0, DType::F64).unwrap();
        assert_eq!(r.shape(), &[3]);
        assert!((r.values()[0] - 1.0).abs() < 1e-12); // 10^0
        assert!((r.values()[1] - 10.0).abs() < 1e-12); // 10^1
        assert!((r.values()[2] - 100.0).abs() < 1e-10); // 10^2
    }

    #[test]
    fn logspace_base2() {
        let r = UFuncArray::logspace(0.0, 3.0, 4, 2.0, DType::F64).unwrap();
        assert!((r.values()[0] - 1.0).abs() < 1e-12);
        assert!((r.values()[3] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn geomspace_basic() {
        let r = UFuncArray::geomspace(1.0, 1000.0, 4, DType::F64).unwrap();
        assert!((r.values()[0] - 1.0).abs() < 1e-12);
        assert!((r.values()[1] - 10.0).abs() < 1e-10);
        assert!((r.values()[2] - 100.0).abs() < 1e-8);
        assert!((r.values()[3] - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn geomspace_zero_error() {
        assert!(UFuncArray::geomspace(0.0, 10.0, 5, DType::F64).is_err());
    }

    #[test]
    fn fromfunction_2d() {
        let r =
            UFuncArray::fromfunction(&[2, 3], DType::F64, |idx| (idx[0] + idx[1]) as f64).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        // [[0,1,2],[1,2,3]]
        assert_eq!(r.values(), &[0.0, 1.0, 2.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn indices_2d() {
        let r = UFuncArray::indices(&[2, 3], DType::I64).unwrap();
        // shape: [2, 2, 3] — first "plane" is row indices, second is col indices
        assert_eq!(r.shape(), &[2, 2, 3]);
        // row indices: [[0,0,0],[1,1,1]]
        assert_eq!(&r.values()[0..6], &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        // col indices: [[0,1,2],[0,1,2]]
        assert_eq!(&r.values()[6..12], &[0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn identity_3() {
        let r = UFuncArray::identity(3, DType::F64).unwrap();
        assert_eq!(r.shape(), &[3, 3]);
        assert_eq!(r.values(), &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn tri_basic() {
        let r = UFuncArray::tri(3, None, 0, DType::F64);
        assert_eq!(r.shape(), &[3, 3]);
        // [[1,0,0],[1,1,0],[1,1,1]]
        assert_eq!(r.values(), &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn tri_positive_k() {
        let r = UFuncArray::tri(3, None, 1, DType::F64);
        // [[1,1,0],[1,1,1],[1,1,1]]
        assert_eq!(r.values(), &[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn diagflat_basic() {
        let v = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let r = v.diagflat(0);
        assert_eq!(r.shape(), &[3, 3]);
        assert_eq!(r.values(), &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn diagflat_offset() {
        let v = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap();
        let r = v.diagflat(1);
        assert_eq!(r.shape(), &[3, 3]);
        assert_eq!(r.values(), &[0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]);
    }

    // ── set operations ──────────────────────────────────────────────────

    #[test]
    fn unique_with_info_all() {
        let arr = UFuncArray::new(vec![6], vec![3.0, 1.0, 2.0, 1.0, 3.0, 2.0], DType::F64).unwrap();
        let (u, idx, inv, cnt) = arr.unique_with_info(true, true, true);
        assert_eq!(u.values(), &[1.0, 2.0, 3.0]);
        // First occurrence indices: 1 is at index 1, 2 at index 2, 3 at index 0
        assert_eq!(idx.unwrap().values(), &[1.0, 2.0, 0.0]);
        // Inverse: reconstruct original from unique[inverse]
        let inv_vals = inv.unwrap();
        assert_eq!(inv_vals.values(), &[2.0, 0.0, 1.0, 0.0, 2.0, 1.0]);
        // Counts: each appears twice
        assert_eq!(cnt.unwrap().values(), &[2.0, 2.0, 2.0]);
    }

    #[test]
    fn unique_with_info_none() {
        let arr = UFuncArray::new(vec![3], vec![5.0, 3.0, 5.0], DType::F64).unwrap();
        let (u, idx, inv, cnt) = arr.unique_with_info(false, false, false);
        assert_eq!(u.values(), &[3.0, 5.0]);
        assert!(idx.is_none());
        assert!(inv.is_none());
        assert!(cnt.is_none());
    }

    #[test]
    fn in1d_basic() {
        let ar1 = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let ar2 = UFuncArray::new(vec![3], vec![2.0, 4.0, 6.0], DType::F64).unwrap();
        let result = ar1.in1d(&ar2);
        assert_eq!(result.values(), &[0.0, 1.0, 0.0, 1.0]);
        assert_eq!(result.dtype, DType::Bool);
    }

    #[test]
    fn isin_preserves_shape() {
        let arr = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let test = UFuncArray::new(vec![2], vec![2.0, 3.0], DType::F64).unwrap();
        let result = arr.isin(&test);
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.values(), &[0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn union1d_basic() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![2.0, 3.0, 4.0], DType::F64).unwrap();
        let r = a.union1d(&b);
        assert_eq!(r.values(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn union1d_disjoint() {
        let a = UFuncArray::new(vec![2], vec![1.0, 3.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2], vec![2.0, 4.0], DType::F64).unwrap();
        let r = a.union1d(&b);
        assert_eq!(r.values(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn intersect1d_basic() {
        let a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![2.0, 4.0, 6.0], DType::F64).unwrap();
        let r = a.intersect1d(&b);
        assert_eq!(r.values(), &[2.0, 4.0]);
    }

    #[test]
    fn intersect1d_empty() {
        let a = UFuncArray::new(vec![2], vec![1.0, 3.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2], vec![2.0, 4.0], DType::F64).unwrap();
        let r = a.intersect1d(&b);
        assert_eq!(r.values(), &[] as &[f64]);
    }

    #[test]
    fn setdiff1d_basic() {
        let a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2], vec![2.0, 4.0], DType::F64).unwrap();
        let r = a.setdiff1d(&b);
        assert_eq!(r.values(), &[1.0, 3.0]);
    }

    #[test]
    fn setdiff1d_no_overlap() {
        let a = UFuncArray::new(vec![2], vec![1.0, 3.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2], vec![2.0, 4.0], DType::F64).unwrap();
        let r = a.setdiff1d(&b);
        assert_eq!(r.values(), &[1.0, 3.0]);
    }

    #[test]
    fn setxor1d_basic() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![2.0, 3.0, 4.0], DType::F64).unwrap();
        let r = a.setxor1d(&b);
        assert_eq!(r.values(), &[1.0, 4.0]);
    }

    #[test]
    fn setxor1d_identical() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let r = a.setxor1d(&b);
        assert_eq!(r.values(), &[] as &[f64]);
    }

    // ── array manipulation batch 2: column_stack, vsplit, hsplit, dsplit, broadcast_arrays, trim_zeros, resize ──

    #[test]
    fn column_stack_basic() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![4.0, 5.0, 6.0], DType::F64).unwrap();
        let r = UFuncArray::column_stack(&[a, b]).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(r.values(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn vsplit_basic() {
        let a = UFuncArray::new(
            vec![4, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            DType::F64,
        )
        .unwrap();
        let parts = a.vsplit(2).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].shape(), &[2, 2]);
        assert_eq!(parts[0].values(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(parts[1].values(), &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn hsplit_basic() {
        let a = UFuncArray::new(
            vec![2, 4],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            DType::F64,
        )
        .unwrap();
        let parts = a.hsplit(2).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].shape(), &[2, 2]);
        assert_eq!(parts[0].values(), &[1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn broadcast_arrays_basic() {
        let a = UFuncArray::new(vec![3, 1], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![1, 3], vec![10.0, 20.0, 30.0], DType::F64).unwrap();
        let result = UFuncArray::broadcast_arrays(&[&a, &b]).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].shape(), &[3, 3]);
        assert_eq!(result[1].shape(), &[3, 3]);
        // a broadcast: [[1,1,1],[2,2,2],[3,3,3]]
        assert_eq!(
            result[0].values(),
            &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]
        );
        // b broadcast: [[10,20,30],[10,20,30],[10,20,30]]
        assert_eq!(
            result[1].values(),
            &[10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
        );
    }

    #[test]
    fn trim_zeros_basic() {
        let a =
            UFuncArray::new(vec![7], vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0], DType::F64).unwrap();
        let r = a.trim_zeros();
        assert_eq!(r.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn trim_zeros_all_zeros() {
        let a = UFuncArray::new(vec![3], vec![0.0, 0.0, 0.0], DType::F64).unwrap();
        let r = a.trim_zeros();
        assert_eq!(r.values(), &[] as &[f64]);
    }

    #[test]
    fn resize_basic() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let r = a.resize(&[2, 3]).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.values(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn resize_shrink() {
        let a = UFuncArray::new(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0], DType::F64).unwrap();
        let r = a.resize(&[2]).unwrap();
        assert_eq!(r.values(), &[1.0, 2.0]);
    }

    // ── polynomial: polyfit, polyder, polyint, roots ──

    #[test]
    fn polyfit_linear() {
        // Fit y = 2x + 1 to exact data
        let x = UFuncArray::new(vec![3], vec![0.0, 1.0, 2.0], DType::F64).unwrap();
        let y = UFuncArray::new(vec![3], vec![1.0, 3.0, 5.0], DType::F64).unwrap();
        let c = UFuncArray::polyfit(&x, &y, 1).unwrap();
        assert_eq!(c.shape(), &[2]);
        assert!((c.values()[0] - 2.0).abs() < 1e-10); // slope
        assert!((c.values()[1] - 1.0).abs() < 1e-10); // intercept
    }

    #[test]
    fn polyfit_quadratic() {
        // Fit y = x^2 to data
        let x = UFuncArray::new(vec![5], vec![0.0, 1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let y = UFuncArray::new(vec![5], vec![0.0, 1.0, 4.0, 9.0, 16.0], DType::F64).unwrap();
        let c = UFuncArray::polyfit(&x, &y, 2).unwrap();
        assert_eq!(c.shape(), &[3]);
        assert!((c.values()[0] - 1.0).abs() < 1e-8); // x^2 coeff
        assert!(c.values()[1].abs() < 1e-8); // x coeff
        assert!(c.values()[2].abs() < 1e-8); // constant
    }

    #[test]
    fn polyder_basic() {
        // d/dx (3x^2 + 2x + 1) = 6x + 2
        let p = UFuncArray::new(vec![3], vec![3.0, 2.0, 1.0], DType::F64).unwrap();
        let dp = p.polyder().unwrap();
        assert_eq!(dp.shape(), &[2]);
        assert_eq!(dp.values(), &[6.0, 2.0]);
    }

    #[test]
    fn polyder_constant() {
        let p = UFuncArray::new(vec![1], vec![5.0], DType::F64).unwrap();
        let dp = p.polyder().unwrap();
        assert_eq!(dp.values(), &[0.0]);
    }

    #[test]
    fn polyint_basic() {
        // integral of 6x + 2 = 3x^2 + 2x + 0
        let p = UFuncArray::new(vec![2], vec![6.0, 2.0], DType::F64).unwrap();
        let ip = p.polyint().unwrap();
        assert_eq!(ip.shape(), &[3]);
        assert!((ip.values()[0] - 3.0).abs() < 1e-12);
        assert!((ip.values()[1] - 2.0).abs() < 1e-12);
        assert_eq!(ip.values()[2], 0.0); // integration constant
    }

    #[test]
    fn roots_linear() {
        // 2x + 6 = 0 -> x = -3
        let p = UFuncArray::new(vec![2], vec![2.0, 6.0], DType::F64).unwrap();
        let r = p.roots().unwrap();
        assert_eq!(r.shape(), &[1]);
        assert!((r.values()[0] - (-3.0)).abs() < 1e-12);
    }

    #[test]
    fn roots_quadratic() {
        // x^2 - 5x + 6 = 0 -> (x-2)(x-3) = 0 -> roots: 3, 2
        let p = UFuncArray::new(vec![3], vec![1.0, -5.0, 6.0], DType::F64).unwrap();
        let r = p.roots().unwrap();
        assert_eq!(r.shape(), &[2]);
        let mut roots = r.values().to_vec();
        roots.sort_by(|a, b| a.total_cmp(b));
        assert!((roots[0] - 2.0).abs() < 1e-12);
        assert!((roots[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn roots_quadratic_complex() {
        // x^2 + 1 = 0 -> complex roots
        let p = UFuncArray::new(vec![3], vec![1.0, 0.0, 1.0], DType::F64).unwrap();
        let r = p.roots().unwrap();
        assert!(r.values()[0].is_nan()); // complex root represented as NaN
    }

    // ── sorting: partition, argpartition, lexsort ──

    #[test]
    fn partition_basic() {
        let a = UFuncArray::new(vec![5], vec![3.0, 1.0, 4.0, 1.0, 5.0], DType::F64).unwrap();
        let r = a.partition(2).unwrap();
        // After partition with kth=2, element at index 2 should be 3.0 (sorted value)
        // Elements before should be <= 3.0, elements after >= 3.0
        assert!(r.values()[0] <= r.values()[2]);
        assert!(r.values()[1] <= r.values()[2]);
        assert!(r.values()[3] >= r.values()[2]);
        assert!(r.values()[4] >= r.values()[2]);
    }

    #[test]
    fn partition_out_of_bounds() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        assert!(a.partition(5).is_err());
    }

    #[test]
    fn argpartition_basic() {
        let a = UFuncArray::new(vec![5], vec![3.0, 1.0, 4.0, 1.0, 5.0], DType::F64).unwrap();
        let r = a.argpartition(2).unwrap();
        // Index at kth position should point to an element that's the kth smallest
        let kth_idx = r.values()[2] as usize;
        let kth_val = a.values()[kth_idx];
        // All elements pointed to by indices before kth should be <= kth_val
        for i in 0..2 {
            let idx = r.values()[i] as usize;
            assert!(a.values()[idx] <= kth_val);
        }
        for i in 3..5 {
            let idx = r.values()[i] as usize;
            assert!(a.values()[idx] >= kth_val);
        }
    }

    #[test]
    fn lexsort_basic() {
        // Sort by last name (primary), then first name (secondary)
        // Keys: first=[1,3,2,1], last=[2,1,1,2]
        // Sort by last (primary) first: last[1]=1, last[2]=1, last[0]=2, last[3]=2
        // Within ties, sort by first: first[2]=2, first[1]=3 for last=1; first[0]=1, first[3]=1 for last=2
        let first = UFuncArray::new(vec![4], vec![1.0, 3.0, 2.0, 1.0], DType::F64).unwrap();
        let last = UFuncArray::new(vec![4], vec![2.0, 1.0, 1.0, 2.0], DType::F64).unwrap();
        let r = UFuncArray::lexsort(&[&first, &last]).unwrap();
        // Expected: [2, 1, 0, 3] (sort by last key first, break ties with first key)
        assert_eq!(r.values(), &[2.0, 1.0, 0.0, 3.0]);
    }

    #[test]
    fn lexsort_single_key() {
        let k = UFuncArray::new(vec![3], vec![3.0, 1.0, 2.0], DType::F64).unwrap();
        let r = UFuncArray::lexsort(&[&k]).unwrap();
        assert_eq!(r.values(), &[1.0, 2.0, 0.0]);
    }

    // ── statistics: quantile, average, cov, corrcoef, digitize, histogram2d ──

    #[test]
    fn quantile_basic() {
        let a = UFuncArray::new(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0], DType::F64).unwrap();
        let r = a.quantile(0.5, None).unwrap();
        assert_eq!(r.values(), &[3.0]);
    }

    #[test]
    fn quantile_quartile() {
        let a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let r = a.quantile(0.25, None).unwrap();
        assert!((r.values()[0] - 1.75).abs() < 1e-12);
    }

    #[test]
    fn average_uniform() {
        let a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let r = a.average(None, None).unwrap();
        assert_eq!(r.values(), &[2.5]);
    }

    #[test]
    fn average_weighted() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let w = UFuncArray::new(vec![3], vec![3.0, 1.0, 1.0], DType::F64).unwrap();
        let r = a.average(Some(&w), None).unwrap();
        // (1*3 + 2*1 + 3*1) / (3+1+1) = 8/5 = 1.6
        assert!((r.values()[0] - 1.6).abs() < 1e-12);
    }

    #[test]
    fn cov_basic() {
        // Two variables, 3 observations each
        let m =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let c = m.cov().unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        // var([1,2,3]) = 1.0 (ddof=1), cov = 1.0 (perfect correlation)
        assert!((c.values()[0] - 1.0).abs() < 1e-12); // var(x)
        assert!((c.values()[1] - 1.0).abs() < 1e-12); // cov(x,y)
        assert!((c.values()[2] - 1.0).abs() < 1e-12); // cov(y,x)
        assert!((c.values()[3] - 1.0).abs() < 1e-12); // var(y)
    }

    #[test]
    fn corrcoef_basic() {
        let m =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let c = m.corrcoef().unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        // Perfect correlation
        assert!((c.values()[0] - 1.0).abs() < 1e-12);
        assert!((c.values()[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn corrcoef_negative() {
        let m =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 6.0, 5.0, 4.0], DType::F64).unwrap();
        let c = m.corrcoef().unwrap();
        // Perfect negative correlation
        assert!((c.values()[1] - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn digitize_basic() {
        let x = UFuncArray::new(vec![4], vec![0.5, 1.5, 2.5, 3.5], DType::F64).unwrap();
        let bins = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let r = x.digitize(&bins).unwrap();
        // 0.5 < 1 => 0, 1 <= 1.5 < 2 => 1, 2 <= 2.5 < 3 => 2, 3.5 >= 3 => 3
        assert_eq!(r.values(), &[0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn histogram2d_basic() {
        let x = UFuncArray::new(vec![4], vec![0.0, 1.0, 2.0, 3.0], DType::F64).unwrap();
        let y = UFuncArray::new(vec![4], vec![0.0, 1.0, 2.0, 3.0], DType::F64).unwrap();
        let (h, xe, ye) = x.histogram2d(&y, 2, 2).unwrap();
        assert_eq!(h.shape(), &[2, 2]);
        assert_eq!(xe.shape(), &[3]); // 3 edges for 2 bins
        assert_eq!(ye.shape(), &[3]);
        // All points on the diagonal, so hist should be [[2,0],[0,2]]
        assert_eq!(h.values()[0] + h.values()[3], 4.0); // diagonal sum
    }

    // ── array manipulation: swapaxes, moveaxis, atleast, dstack, delete, insert, append, rot90 ──

    #[test]
    fn swapaxes_basic() {
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let r = a.swapaxes(0, 1).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(r.values(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn moveaxis_basic() {
        let a = UFuncArray::new(
            vec![2, 3, 4],
            (0..24).map(|i| i as f64).collect(),
            DType::F64,
        )
        .unwrap();
        let r = a.moveaxis(0, 2).unwrap();
        assert_eq!(r.shape(), &[3, 4, 2]);
    }

    #[test]
    fn atleast_1d_scalar() {
        let s = UFuncArray::scalar(5.0, DType::F64);
        let r = s.atleast_1d();
        assert_eq!(r.shape(), &[1]);
        assert_eq!(r.values(), &[5.0]);
    }

    #[test]
    fn atleast_2d_1d() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let r = a.atleast_2d();
        assert_eq!(r.shape(), &[1, 3]);
    }

    #[test]
    fn atleast_3d_2d() {
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let r = a.atleast_3d();
        assert_eq!(r.shape(), &[2, 3, 1]);
    }

    #[test]
    fn dstack_basic() {
        let a = UFuncArray::new(vec![2, 1], vec![1.0, 2.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2, 1], vec![3.0, 4.0], DType::F64).unwrap();
        let r = UFuncArray::dstack(&[a, b]).unwrap();
        assert_eq!(r.shape(), &[2, 1, 2]);
        assert_eq!(r.values(), &[1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn delete_no_axis() {
        let a = UFuncArray::new(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0], DType::F64).unwrap();
        let r = a.delete(&[1, 3], None).unwrap();
        assert_eq!(r.values(), &[1.0, 3.0, 5.0]);
    }

    #[test]
    fn delete_axis_0() {
        let a =
            UFuncArray::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let r = a.delete(&[1], Some(0)).unwrap();
        assert_eq!(r.shape(), &[2, 2]);
        assert_eq!(r.values(), &[1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn insert_no_axis() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let ins = UFuncArray::new(vec![2], vec![10.0, 20.0], DType::F64).unwrap();
        let r = a.insert(1, &ins, None).unwrap();
        assert_eq!(r.values(), &[1.0, 10.0, 20.0, 2.0, 3.0]);
    }

    #[test]
    fn append_no_axis() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2], vec![4.0, 5.0], DType::F64).unwrap();
        let r = a.append(&b, None).unwrap();
        assert_eq!(r.values(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn append_axis_0() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![1, 2], vec![5.0, 6.0], DType::F64).unwrap();
        let r = a.append(&b, Some(0)).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(r.values(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn rot90_basic() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let r = a.rot90(1).unwrap();
        assert_eq!(r.shape(), &[2, 2]);
        // np.rot90([[1,2],[3,4]]) => [[2,4],[1,3]]
        assert_eq!(r.values(), &[2.0, 4.0, 1.0, 3.0]);
    }

    #[test]
    fn rot90_twice() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let r = a.rot90(2).unwrap();
        // np.rot90(a, 2) => [[4,3],[2,1]]
        assert_eq!(r.values(), &[4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn rot90_negative() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        // rot90(k=-1) is same as rot90(k=3)
        let r = a.rot90(-1).unwrap();
        // np.rot90(a, 3) => [[3,1],[4,2]]
        assert_eq!(r.values(), &[3.0, 1.0, 4.0, 2.0]);
    }

    // ── NaN-aware reductions ────────────────────────────────────────────

    #[test]
    fn nansum_basic() {
        let a = UFuncArray::new(vec![4], vec![1.0, f64::NAN, 3.0, 4.0], DType::F64).unwrap();
        let r = a.nansum(None, false).unwrap();
        assert_eq!(r.values(), &[8.0]);
    }

    #[test]
    fn nansum_axis() {
        let a = UFuncArray::new(
            vec![2, 3],
            vec![1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0],
            DType::F64,
        )
        .unwrap();
        let r = a.nansum(Some(1), false).unwrap();
        assert_eq!(r.values(), &[4.0, 15.0]);
    }

    #[test]
    fn nanmean_basic() {
        let a = UFuncArray::new(vec![4], vec![1.0, f64::NAN, 3.0, 4.0], DType::F64).unwrap();
        let r = a.nanmean(None, false).unwrap();
        let v = r.values()[0];
        assert!((v - 8.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn nanmean_axis() {
        let a = UFuncArray::new(
            vec![2, 3],
            vec![1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0],
            DType::F64,
        )
        .unwrap();
        let r = a.nanmean(Some(1), false).unwrap();
        assert!((r.values()[0] - 2.0).abs() < 1e-12); // (1+3)/2
        assert!((r.values()[1] - 5.0).abs() < 1e-12); // (4+5+6)/3
    }

    #[test]
    fn nanvar_basic() {
        let a = UFuncArray::new(vec![5], vec![1.0, f64::NAN, 3.0, 4.0, 2.0], DType::F64).unwrap();
        let r = a.nanvar(None, false, 0).unwrap();
        // values: 1,3,4,2  mean=2.5, var = ((1-2.5)^2 + (3-2.5)^2 + (4-2.5)^2 + (2-2.5)^2)/4
        //       = (2.25 + 0.25 + 2.25 + 0.25)/4 = 5.0/4 = 1.25
        assert!((r.values()[0] - 1.25).abs() < 1e-12);
    }

    #[test]
    fn nanstd_basic() {
        let a = UFuncArray::new(vec![5], vec![1.0, f64::NAN, 3.0, 4.0, 2.0], DType::F64).unwrap();
        let r = a.nanstd(None, false, 0).unwrap();
        assert!((r.values()[0] - 1.25_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn nanmin_basic() {
        let a = UFuncArray::new(vec![4], vec![f64::NAN, 3.0, 1.0, 2.0], DType::F64).unwrap();
        let r = a.nanmin(None, false).unwrap();
        assert_eq!(r.values(), &[1.0]);
    }

    #[test]
    fn nanmax_basic() {
        let a = UFuncArray::new(vec![4], vec![f64::NAN, 3.0, 1.0, 2.0], DType::F64).unwrap();
        let r = a.nanmax(None, false).unwrap();
        assert_eq!(r.values(), &[3.0]);
    }

    #[test]
    fn nanmin_axis() {
        let a = UFuncArray::new(
            vec![2, 3],
            vec![f64::NAN, 2.0, 3.0, 4.0, f64::NAN, 6.0],
            DType::F64,
        )
        .unwrap();
        let r = a.nanmin(Some(1), false).unwrap();
        assert_eq!(r.values(), &[2.0, 4.0]);
    }

    #[test]
    fn nanmax_axis() {
        let a = UFuncArray::new(
            vec![2, 3],
            vec![f64::NAN, 2.0, 3.0, 4.0, f64::NAN, 6.0],
            DType::F64,
        )
        .unwrap();
        let r = a.nanmax(Some(1), false).unwrap();
        assert_eq!(r.values(), &[3.0, 6.0]);
    }

    #[test]
    fn nanmedian_basic() {
        let a =
            UFuncArray::new(vec![5], vec![1.0, f64::NAN, 3.0, 2.0, f64::NAN], DType::F64).unwrap();
        let r = a.nanmedian(None).unwrap();
        assert_eq!(r.values(), &[2.0]); // median of [1,2,3]
    }

    #[test]
    fn nanargmin_basic() {
        let a = UFuncArray::new(vec![4], vec![f64::NAN, 3.0, 1.0, 2.0], DType::F64).unwrap();
        let r = a.nanargmin(None).unwrap();
        assert_eq!(r.values(), &[2.0]); // index 2 has value 1.0
    }

    #[test]
    fn nanargmax_basic() {
        let a = UFuncArray::new(vec![4], vec![f64::NAN, 3.0, 1.0, 2.0], DType::F64).unwrap();
        let r = a.nanargmax(None).unwrap();
        assert_eq!(r.values(), &[1.0]); // index 1 has value 3.0
    }

    #[test]
    fn nanargmin_axis() {
        let a = UFuncArray::new(
            vec![2, 3],
            vec![f64::NAN, 2.0, 3.0, 4.0, f64::NAN, 1.0],
            DType::F64,
        )
        .unwrap();
        let r = a.nanargmin(Some(1)).unwrap();
        assert_eq!(r.values(), &[1.0, 2.0]); // row0: min at col1, row1: min at col2
    }

    #[test]
    fn nanargmax_axis() {
        let a = UFuncArray::new(
            vec![2, 3],
            vec![f64::NAN, 2.0, 3.0, 4.0, f64::NAN, 1.0],
            DType::F64,
        )
        .unwrap();
        let r = a.nanargmax(Some(1)).unwrap();
        assert_eq!(r.values(), &[2.0, 0.0]); // row0: max at col2, row1: max at col0
    }

    // ── window function tests ────────

    #[test]
    fn hamming_window() {
        let w = UFuncArray::hamming(5);
        assert_eq!(w.shape(), &[5]);
        // hamming(5) endpoints: 0.08, midpoint: 1.0
        assert!((w.values()[0] - 0.08).abs() < 1e-10);
        assert!((w.values()[2] - 1.0).abs() < 1e-10);
        assert!((w.values()[4] - 0.08).abs() < 1e-10);
        // Symmetry
        assert!((w.values()[1] - w.values()[3]).abs() < 1e-10);
    }

    #[test]
    fn hamming_edge_cases() {
        let w0 = UFuncArray::hamming(0);
        assert_eq!(w0.shape(), &[0]);
        let w1 = UFuncArray::hamming(1);
        assert_eq!(w1.values(), &[1.0]);
    }

    #[test]
    fn hanning_window() {
        let w = UFuncArray::hanning(5);
        assert_eq!(w.shape(), &[5]);
        // hanning(5) endpoints: 0.0, midpoint: 1.0
        assert!(w.values()[0].abs() < 1e-10);
        assert!((w.values()[2] - 1.0).abs() < 1e-10);
        assert!(w.values()[4].abs() < 1e-10);
        assert!((w.values()[1] - w.values()[3]).abs() < 1e-10);
    }

    #[test]
    fn blackman_window() {
        let w = UFuncArray::blackman(5);
        assert_eq!(w.shape(), &[5]);
        // blackman(5) endpoints very close to 0, midpoint: 1.0
        assert!(w.values()[0].abs() < 1e-10);
        assert!((w.values()[2] - 1.0).abs() < 1e-10);
        assert!(w.values()[4].abs() < 1e-10);
        assert!((w.values()[1] - w.values()[3]).abs() < 1e-10);
    }

    #[test]
    fn bartlett_window() {
        let w = UFuncArray::bartlett(5);
        assert_eq!(w.shape(), &[5]);
        // bartlett(5): [0.0, 0.5, 1.0, 0.5, 0.0]
        assert!(w.values()[0].abs() < 1e-10);
        assert!((w.values()[1] - 0.5).abs() < 1e-10);
        assert!((w.values()[2] - 1.0).abs() < 1e-10);
        assert!((w.values()[3] - 0.5).abs() < 1e-10);
        assert!(w.values()[4].abs() < 1e-10);
    }

    #[test]
    fn kaiser_window() {
        let w = UFuncArray::kaiser(5, 14.0);
        assert_eq!(w.shape(), &[5]);
        // Kaiser window: midpoint is 1.0, endpoints are small
        assert!((w.values()[2] - 1.0).abs() < 1e-10);
        assert!(w.values()[0] < 0.1); // endpoints much smaller
        assert!((w.values()[0] - w.values()[4]).abs() < 1e-10); // symmetric
        assert!((w.values()[1] - w.values()[3]).abs() < 1e-10);
    }

    #[test]
    fn kaiser_beta_zero() {
        // beta=0 gives rectangular window (all ones)
        let w = UFuncArray::kaiser(5, 0.0);
        for &v in w.values() {
            assert!((v - 1.0).abs() < 1e-10);
        }
    }

    // ── advanced indexing tests ────────

    #[test]
    fn take_along_axis_basic() {
        // 2x3 array, pick indices along axis 1
        let a = UFuncArray::new(
            vec![2, 3],
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            DType::F64,
        )
        .unwrap();
        let idx = UFuncArray::new(vec![2, 1], vec![2.0, 0.0], DType::I64).unwrap();
        let r = a.take_along_axis(&idx, 1).unwrap();
        assert_eq!(r.shape(), &[2, 1]);
        assert_eq!(r.values(), &[30.0, 40.0]);
    }

    #[test]
    fn put_along_axis_basic() {
        let mut a = UFuncArray::new(
            vec![2, 3],
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            DType::F64,
        )
        .unwrap();
        let idx = UFuncArray::new(vec![2, 1], vec![1.0, 2.0], DType::I64).unwrap();
        let vals = UFuncArray::new(vec![2, 1], vec![99.0, 88.0], DType::F64).unwrap();
        a.put_along_axis(&idx, &vals, 1).unwrap();
        assert_eq!(a.values(), &[10.0, 99.0, 30.0, 40.0, 50.0, 88.0]);
    }

    #[test]
    fn extract_basic() {
        let cond = UFuncArray::new(vec![5], vec![1.0, 0.0, 1.0, 0.0, 1.0], DType::Bool).unwrap();
        let arr = UFuncArray::new(vec![5], vec![10.0, 20.0, 30.0, 40.0, 50.0], DType::F64).unwrap();
        let r = UFuncArray::extract(&cond, &arr).unwrap();
        assert_eq!(r.values(), &[10.0, 30.0, 50.0]);
    }

    #[test]
    fn place_cyclic() {
        let mut a = UFuncArray::new(vec![4], vec![0.0, 0.0, 0.0, 0.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![4], vec![1.0, 0.0, 1.0, 1.0], DType::Bool).unwrap();
        a.place(&mask, &[10.0, 20.0]).unwrap();
        assert_eq!(a.values(), &[10.0, 0.0, 20.0, 10.0]); // cycles back to 10.0
    }

    #[test]
    fn put_basic() {
        let mut a = UFuncArray::new(vec![5], vec![0.0; 5], DType::F64).unwrap();
        a.put(&[0, 2, 4], &[10.0, 20.0, 30.0]).unwrap();
        assert_eq!(a.values(), &[10.0, 0.0, 20.0, 0.0, 30.0]);
    }

    #[test]
    fn fill_diagonal_basic() {
        let mut a = UFuncArray::new(vec![3, 3], vec![0.0; 9], DType::F64).unwrap();
        a.fill_diagonal(5.0).unwrap();
        assert_eq!(a.values(), &[5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0]);
    }

    #[test]
    fn fill_diagonal_rect() {
        let mut a = UFuncArray::new(vec![2, 4], vec![0.0; 8], DType::F64).unwrap();
        a.fill_diagonal(1.0).unwrap();
        assert_eq!(a.values(), &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn diag_indices_basic() {
        let (arrays, dtype) = UFuncArray::diag_indices(3, 2);
        assert_eq!(arrays.len(), 2);
        assert_eq!(dtype, DType::I64);
        assert_eq!(arrays[0].values(), &[0.0, 1.0, 2.0]);
        assert_eq!(arrays[1].values(), &[0.0, 1.0, 2.0]);
    }

    #[test]
    fn tril_indices_basic() {
        let (rows, cols) = UFuncArray::tril_indices(3, 3, 0);
        assert_eq!(rows.values(), &[0.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        assert_eq!(cols.values(), &[0.0, 0.0, 1.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn triu_indices_basic() {
        let (rows, cols) = UFuncArray::triu_indices(3, 3, 0);
        assert_eq!(rows.values(), &[0.0, 0.0, 0.0, 1.0, 1.0, 2.0]);
        assert_eq!(cols.values(), &[0.0, 1.0, 2.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn triu_indices_offset() {
        let (rows, cols) = UFuncArray::triu_indices(3, 3, 1);
        // k=1: strictly above diagonal
        assert_eq!(rows.values(), &[0.0, 0.0, 1.0]);
        assert_eq!(cols.values(), &[1.0, 2.0, 2.0]);
    }

    // ── nanprod / nancumsum / nancumprod tests ────────

    #[test]
    fn nanprod_basic() {
        let a = UFuncArray::new(vec![4], vec![1.0, f64::NAN, 3.0, 4.0], DType::F64).unwrap();
        let r = a.nanprod(None, false).unwrap();
        assert!((r.values()[0] - 12.0).abs() < 1e-12); // 1*3*4
    }

    #[test]
    fn nanprod_axis() {
        let a = UFuncArray::new(
            vec![2, 3],
            vec![1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0],
            DType::F64,
        )
        .unwrap();
        let r = a.nanprod(Some(1), false).unwrap();
        assert!((r.values()[0] - 3.0).abs() < 1e-12); // 1*1*3 (NaN→1)
        assert!((r.values()[1] - 120.0).abs() < 1e-12); // 4*5*6
    }

    #[test]
    fn nancumsum_basic() {
        let a = UFuncArray::new(vec![4], vec![1.0, f64::NAN, 3.0, 4.0], DType::F64).unwrap();
        let r = a.nancumsum(None).unwrap();
        assert_eq!(r.values(), &[1.0, 1.0, 4.0, 8.0]);
    }

    #[test]
    fn nancumprod_basic() {
        let a = UFuncArray::new(vec![4], vec![2.0, f64::NAN, 3.0, 4.0], DType::F64).unwrap();
        let r = a.nancumprod(None).unwrap();
        assert_eq!(r.values(), &[2.0, 2.0, 6.0, 24.0]);
    }

    // ── pad mode tests ────────

    #[test]
    fn pad_edge_1d() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let r = a.pad_edge(&[(2, 2)]).unwrap();
        assert_eq!(r.values(), &[1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn pad_edge_2d() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let r = a.pad_edge(&[(1, 1), (1, 1)]).unwrap();
        assert_eq!(r.shape(), &[4, 4]);
        // top-left corner should be 1.0, bottom-right should be 4.0
        assert_eq!(r.values()[0], 1.0);
        assert_eq!(r.values()[15], 4.0);
    }

    #[test]
    fn pad_wrap_1d() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let r = a.pad_wrap(&[(2, 2)]).unwrap();
        // wrap: [2, 3, 1, 2, 3, 1, 2]
        assert_eq!(r.values(), &[2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]);
    }

    #[test]
    fn pad_reflect_1d() {
        let a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let r = a.pad_reflect(&[(2, 2)]).unwrap();
        // reflect (no edge dup): [3, 2, 1, 2, 3, 4, 3, 2]
        assert_eq!(r.values(), &[3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
    }

    #[test]
    fn pad_symmetric_1d() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let r = a.pad_symmetric(&[(2, 2)]).unwrap();
        // symmetric (edge dup): [2, 1, 1, 2, 3, 3, 2]
        assert_eq!(r.values(), &[2.0, 1.0, 1.0, 2.0, 3.0, 3.0, 2.0]);
    }

    // ── index utility tests ────────

    #[test]
    fn unravel_index_basic() {
        let indices = UFuncArray::new(vec![3], vec![0.0, 4.0, 7.0], DType::I64).unwrap();
        let result = UFuncArray::unravel_index(&indices, &[2, 4]).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].values(), &[0.0, 1.0, 1.0]); // row indices
        assert_eq!(result[1].values(), &[0.0, 0.0, 3.0]); // col indices
    }

    #[test]
    fn ravel_multi_index_basic() {
        let rows = UFuncArray::new(vec![3], vec![0.0, 1.0, 1.0], DType::I64).unwrap();
        let cols = UFuncArray::new(vec![3], vec![0.0, 0.0, 3.0], DType::I64).unwrap();
        let r = UFuncArray::ravel_multi_index(&[&rows, &cols], &[2, 4]).unwrap();
        assert_eq!(r.values(), &[0.0, 4.0, 7.0]);
    }

    #[test]
    fn unravel_ravel_roundtrip() {
        let indices = UFuncArray::new(vec![4], vec![0.0, 5.0, 11.0, 23.0], DType::I64).unwrap();
        let shape = [2, 3, 4];
        let coords = UFuncArray::unravel_index(&indices, &shape).unwrap();
        let refs: Vec<&UFuncArray> = coords.iter().collect();
        let flat = UFuncArray::ravel_multi_index(&refs, &shape).unwrap();
        assert_eq!(flat.values(), indices.values());
    }

    #[test]
    fn ogrid_basic() {
        let grids = UFuncArray::ogrid(&[(0.0, 2.0, 3), (0.0, 4.0, 5)]);
        assert_eq!(grids.len(), 2);
        assert_eq!(grids[0].shape(), &[3, 1]);
        assert_eq!(grids[1].shape(), &[1, 5]);
        assert_eq!(grids[0].values(), &[0.0, 1.0, 2.0]);
        assert_eq!(grids[1].values(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn mgrid_basic() {
        let grids = UFuncArray::mgrid(&[(0.0, 1.0, 2), (0.0, 1.0, 2)]);
        assert_eq!(grids.len(), 2);
        assert_eq!(grids[0].shape(), &[2, 2]);
        assert_eq!(grids[1].shape(), &[2, 2]);
        // First grid varies along axis 0: [[0,0],[1,1]]
        assert_eq!(grids[0].values(), &[0.0, 0.0, 1.0, 1.0]);
        // Second grid varies along axis 1: [[0,1],[0,1]]
        assert_eq!(grids[1].values(), &[0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn broadcast_to_basic() {
        let a = UFuncArray::new(vec![1, 3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let r = a.broadcast_to(&[3, 3]).unwrap();
        assert_eq!(r.shape(), &[3, 3]);
        assert_eq!(r.values(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn broadcast_to_scalar() {
        let a = UFuncArray::scalar(5.0, DType::F64);
        let r = a.broadcast_to(&[2, 3]).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.values(), &[5.0; 6]);
    }

    // ── tensor operation tests ────────

    #[test]
    fn kron_basic() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2, 2], vec![0.0, 5.0, 6.0, 7.0], DType::F64).unwrap();
        let r = a.kron(&b).unwrap();
        assert_eq!(r.shape(), &[4, 4]);
        // First block: 1*b = [[0,5],[6,7]]
        assert_eq!(r.values()[0], 0.0);
        assert_eq!(r.values()[1], 5.0);
        // Second block: 2*b
        assert_eq!(r.values()[2], 0.0);
        assert_eq!(r.values()[3], 10.0);
    }

    #[test]
    fn tensordot_matrix_multiply() {
        // tensordot(a, b, axes=1) should be matrix multiply for 2-D
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let b = UFuncArray::new(
            vec![3, 2],
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            DType::F64,
        )
        .unwrap();
        let r = a.tensordot(&b, 1).unwrap();
        assert_eq!(r.shape(), &[2, 2]);
        // [1*7+2*9+3*11, 1*8+2*10+3*12, 4*7+5*9+6*11, 4*8+5*10+6*12]
        assert_eq!(r.values(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn tensordot_full_contraction() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![4.0, 5.0, 6.0], DType::F64).unwrap();
        let r = a.tensordot(&b, 1).unwrap();
        assert_eq!(r.values(), &[32.0]); // 1*4+2*5+3*6=32
    }

    #[test]
    fn vdot_basic() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0], DType::F64).unwrap();
        let r = a.vdot(&b).unwrap();
        assert_eq!(r.values(), &[70.0]); // 1*5+2*6+3*7+4*8=70
    }

    #[test]
    fn multi_dot_basic() {
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let b =
            UFuncArray::new(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0], DType::F64).unwrap();
        let c = UFuncArray::new(vec![2, 1], vec![1.0, 2.0], DType::F64).unwrap();
        let r = UFuncArray::multi_dot(&[&a, &b, &c]).unwrap();
        assert_eq!(r.shape(), &[2, 1]);
    }

    // ── polynomial extension tests ────────

    #[test]
    fn poly_from_roots() {
        // roots [1, 2, 3] -> (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
        let roots = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let coeffs = UFuncArray::poly(&roots).unwrap();
        assert_eq!(coeffs.shape(), &[4]);
        assert!((coeffs.values()[0] - 1.0).abs() < 1e-10);
        assert!((coeffs.values()[1] - (-6.0)).abs() < 1e-10);
        assert!((coeffs.values()[2] - 11.0).abs() < 1e-10);
        assert!((coeffs.values()[3] - (-6.0)).abs() < 1e-10);
    }

    #[test]
    fn polymul_basic() {
        // (x + 1) * (x - 1) = x^2 - 1 → [1, 0, -1]
        let a = UFuncArray::new(vec![2], vec![1.0, 1.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2], vec![1.0, -1.0], DType::F64).unwrap();
        let r = a.polymul(&b).unwrap();
        assert_eq!(r.values(), &[1.0, 0.0, -1.0]);
    }

    #[test]
    fn polyadd_basic() {
        // (x^2 + 1) + (2x + 3) = x^2 + 2x + 4
        let a = UFuncArray::new(vec![3], vec![1.0, 0.0, 1.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2], vec![2.0, 3.0], DType::F64).unwrap();
        let r = a.polyadd(&b).unwrap();
        assert_eq!(r.values(), &[1.0, 2.0, 4.0]);
    }

    #[test]
    fn polysub_basic() {
        let a = UFuncArray::new(vec![3], vec![1.0, 0.0, 1.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![0.0, 1.0, 1.0], DType::F64).unwrap();
        let r = a.polysub(&b).unwrap();
        assert_eq!(r.values(), &[1.0, -1.0, 0.0]);
    }

    #[test]
    fn polydiv_basic() {
        // (x^2 - 1) / (x - 1) = (x + 1) remainder 0
        let a = UFuncArray::new(vec![3], vec![1.0, 0.0, -1.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2], vec![1.0, -1.0], DType::F64).unwrap();
        let (q, r) = a.polydiv(&b).unwrap();
        assert_eq!(q.shape(), &[2]);
        assert!((q.values()[0] - 1.0).abs() < 1e-10);
        assert!((q.values()[1] - 1.0).abs() < 1e-10);
        assert!(r.values()[0].abs() < 1e-10);
    }

    #[test]
    fn polydiv_with_remainder() {
        // (x^2 + 1) / (x + 1) = x - 1 remainder 2
        let a = UFuncArray::new(vec![3], vec![1.0, 0.0, 1.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2], vec![1.0, 1.0], DType::F64).unwrap();
        let (q, r) = a.polydiv(&b).unwrap();
        assert!((q.values()[0] - 1.0).abs() < 1e-10);
        assert!((q.values()[1] - (-1.0)).abs() < 1e-10);
        assert!((r.values()[0] - 2.0).abs() < 1e-10);
    }

    // ── gradient N-D, piecewise, apply_along_axis tests ────────

    #[test]
    fn gradient_1d_compat() {
        let a = UFuncArray::new(vec![4], vec![1.0, 3.0, 6.0, 10.0], DType::F64).unwrap();
        let r = a.gradient().unwrap();
        assert_eq!(r.shape(), &[4]);
        assert_eq!(r.values()[0], 2.0); // forward: 3-1
        assert_eq!(r.values()[1], 2.5); // central: (6-1)/2
        assert_eq!(r.values()[2], 3.5); // central: (10-3)/2
        assert_eq!(r.values()[3], 4.0); // backward: 10-6
    }

    #[test]
    fn gradient_2d_axis0() {
        // 2x3: [[1,2,3],[4,5,6]]
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let r = a.gradient_axis(Some(0)).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        // Along axis 0: [4-1, 5-2, 6-3] for both rows (forward/backward only with 2 elements)
        assert_eq!(r.values(), &[3.0, 3.0, 3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn gradient_2d_axis1() {
        let a = UFuncArray::new(
            vec![2, 3],
            vec![1.0, 3.0, 6.0, 10.0, 15.0, 21.0],
            DType::F64,
        )
        .unwrap();
        let r = a.gradient_axis(Some(1)).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        // Row 0: forward=2, central=2.5, backward=3
        assert_eq!(r.values()[0], 2.0);
        assert_eq!(r.values()[1], 2.5);
        assert_eq!(r.values()[2], 3.0);
    }

    #[test]
    fn piecewise_basic() {
        let x = UFuncArray::new(vec![6], vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], DType::F64).unwrap();
        let cond_neg =
            UFuncArray::new(vec![6], vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0], DType::Bool).unwrap();
        let cond_pos =
            UFuncArray::new(vec![6], vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], DType::Bool).unwrap();
        // funclist: [-1.0 for neg, 1.0 for pos, 0.0 default]
        let r = x
            .piecewise(&[cond_neg, cond_pos], &[-1.0, 1.0, 0.0])
            .unwrap();
        assert_eq!(r.values(), &[-1.0, -1.0, 0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn apply_along_axis_basic() {
        // Sum each column of a 2x3 array
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let r = a
            .apply_along_axis(|slice| slice.reduce_sum(None, false), 0)
            .unwrap();
        // After applying sum along axis 0, each column sums to a scalar
        assert_eq!(r.shape(), &[1, 3]);
        assert_eq!(r.values(), &[5.0, 7.0, 9.0]);
    }

    // ── formatting tests ────────

    #[test]
    fn array2string_1d() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let opts = PrintOptions::default();
        let s = a.array2string(&opts);
        assert!(s.starts_with('['));
        assert!(s.ends_with(']'));
        assert!(s.contains("1."));
    }

    #[test]
    fn array2string_2d() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let s = a.array2string(&PrintOptions::default());
        assert!(s.starts_with("[["));
    }

    #[test]
    fn array2string_summarized() {
        let vals: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let a = UFuncArray::new(vec![100], vals, DType::F64).unwrap();
        let opts = PrintOptions {
            threshold: 10,
            edgeitems: 3,
            ..PrintOptions::default()
        };
        let s = a.array2string(&opts);
        assert!(s.contains("..."));
    }

    #[test]
    fn array_repr_basic() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let r = a.array_repr(&PrintOptions::default());
        assert!(r.starts_with("array("));
        assert!(r.ends_with(')'));
    }

    #[test]
    fn array_repr_with_dtype() {
        let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::I32).unwrap();
        let r = a.array_repr(&PrintOptions::default());
        assert!(r.contains("dtype=int32"));
    }

    #[test]
    fn array2string_scalar() {
        let a = UFuncArray::scalar(42.0, DType::F64);
        let s = a.array2string(&PrintOptions::default());
        assert!(s.contains("42."));
    }

    // ── misc math tests ────────

    #[test]
    fn vander_basic() {
        let x = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let v = x.vander(None, false).unwrap();
        assert_eq!(v.shape(), &[3, 3]);
        // Row for x=2: [4, 2, 1]
        assert_eq!(v.values()[3], 4.0);
        assert_eq!(v.values()[4], 2.0);
        assert_eq!(v.values()[5], 1.0);
    }

    #[test]
    fn vander_increasing() {
        let x = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let v = x.vander(None, true).unwrap();
        // Row for x=2: [1, 2, 4]
        assert_eq!(v.values()[3], 1.0);
        assert_eq!(v.values()[4], 2.0);
        assert_eq!(v.values()[5], 4.0);
    }

    #[test]
    fn ediff1d_basic() {
        let a = UFuncArray::new(vec![4], vec![1.0, 3.0, 6.0, 10.0], DType::F64).unwrap();
        let r = a.ediff1d().unwrap();
        assert_eq!(r.values(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn trapezoid_basic() {
        // trapezoid([1, 2, 3], dx=1) = 0.5*(1+2)*1 + 0.5*(2+3)*1 = 1.5 + 2.5 = 4.0
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let r = a.trapezoid(1.0).unwrap();
        assert!((r.values()[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn sinc_basic() {
        let a = UFuncArray::new(vec![3], vec![0.0, 1.0, -1.0], DType::F64).unwrap();
        let r = a.sinc();
        assert!((r.values()[0] - 1.0).abs() < 1e-10); // sinc(0)=1
        assert!(r.values()[1].abs() < 1e-10); // sinc(1)≈0
        assert!(r.values()[2].abs() < 1e-10); // sinc(-1)≈0
    }

    #[test]
    fn i0_basic() {
        let a = UFuncArray::new(vec![2], vec![0.0, 1.0], DType::F64).unwrap();
        let r = a.i0();
        assert!((r.values()[0] - 1.0).abs() < 1e-6); // I0(0) = 1
        assert!((r.values()[1] - 1.2660658).abs() < 1e-4); // I0(1) ≈ 1.2660658
    }

    #[test]
    fn unwrap_basic() {
        // Phase wrapping: [0, pi/2, pi, pi/2 - 2pi, 0 - 2pi]
        let pi = std::f64::consts::PI;
        let a = UFuncArray::new(
            vec![5],
            vec![0.0, pi / 2.0, pi, pi / 2.0 - 2.0 * pi, -2.0 * pi],
            DType::F64,
        )
        .unwrap();
        let r = a.unwrap(None).unwrap();
        // After unwrap, should be monotonically reasonable
        assert_eq!(r.shape(), &[5]);
        // The first two values should be unchanged
        assert!((r.values()[0]).abs() < 1e-10);
        assert!((r.values()[1] - pi / 2.0).abs() < 1e-10);
    }

    // ── histogram enhancement tests ────────

    #[test]
    fn histogram_edges_basic() {
        let data =
            UFuncArray::new(vec![6], vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5], DType::F64).unwrap();
        let edges = UFuncArray::new(vec![4], vec![0.0, 2.0, 4.0, 6.0], DType::F64).unwrap();
        let (counts, returned_edges) = data.histogram_edges(&edges).unwrap();
        assert_eq!(counts.shape(), &[3]);
        assert_eq!(counts.values(), &[2.0, 2.0, 2.0]);
        assert_eq!(returned_edges.values(), edges.values());
    }

    #[test]
    fn histogram_edges_uneven() {
        let data = UFuncArray::new(vec![4], vec![0.5, 1.5, 5.0, 9.0], DType::F64).unwrap();
        let edges = UFuncArray::new(vec![3], vec![0.0, 2.0, 10.0], DType::F64).unwrap();
        let (counts, _) = data.histogram_edges(&edges).unwrap();
        assert_eq!(counts.values(), &[2.0, 2.0]);
    }

    #[test]
    fn histogram_auto_sturges() {
        let data =
            UFuncArray::new(vec![100], (0..100).map(|i| i as f64).collect(), DType::F64).unwrap();
        let (counts, edges) = data.histogram_auto("sturges").unwrap();
        assert!(counts.shape()[0] > 0);
        assert_eq!(edges.shape()[0], counts.shape()[0] + 1);
    }

    #[test]
    fn histogram_auto_sqrt() {
        let data =
            UFuncArray::new(vec![100], (0..100).map(|i| i as f64).collect(), DType::F64).unwrap();
        let (counts, _) = data.histogram_auto("sqrt").unwrap();
        assert_eq!(counts.shape()[0], 10); // sqrt(100)=10
    }

    #[test]
    fn ndenumerate_2d() {
        let a = UFuncArray::new(
            vec![2, 3],
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            DType::F64,
        )
        .unwrap();
        let pairs = a.ndenumerate();
        assert_eq!(pairs.len(), 6);
        assert_eq!(pairs[0], (vec![0, 0], 10.0));
        assert_eq!(pairs[2], (vec![0, 2], 30.0));
        assert_eq!(pairs[5], (vec![1, 2], 60.0));
    }

    #[test]
    fn ndindex_shape() {
        let indices = UFuncArray::ndindex(&[2, 3]);
        assert_eq!(indices.len(), 6);
        assert_eq!(indices[0], vec![0, 0]);
        assert_eq!(indices[1], vec![0, 1]);
        assert_eq!(indices[3], vec![1, 0]);
        assert_eq!(indices[5], vec![1, 2]);
    }

    #[test]
    fn flat_returns_c_order_slice() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        assert_eq!(a.flat(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn moment_order_2_is_variance() {
        let a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let m2 = a.moment(2).unwrap();
        // variance = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) / 4 = 5/4 = 1.25
        assert!((m2 - 1.25).abs() < 1e-12);
    }

    #[test]
    fn skew_symmetric_distribution() {
        // Symmetric distribution around 0 should have skew ≈ 0
        let a = UFuncArray::new(vec![4], vec![-2.0, -1.0, 1.0, 2.0], DType::F64).unwrap();
        let s = a.skew().unwrap();
        assert!(s.abs() < 1e-12, "skew={s}");
    }

    #[test]
    fn kurtosis_normal_like() {
        // For uniform distribution [1..n], excess kurtosis ≈ -1.2
        let vals: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let a = UFuncArray::new(vec![100], vals, DType::F64).unwrap();
        let k = a.kurtosis().unwrap();
        assert!((k - (-1.2)).abs() < 0.01, "kurtosis={k}");
    }

    #[test]
    fn fft_dc_signal() {
        // FFT of [1, 1, 1, 1] → [4, 0, 0, 0] (DC component only)
        let a = UFuncArray::new(vec![4], vec![1.0, 1.0, 1.0, 1.0], DType::F64).unwrap();
        let f = a.fft(None).unwrap();
        assert_eq!(f.shape, vec![4, 2]);
        // f.values = [re0, im0, re1, im1, re2, im2, re3, im3]
        assert!((f.values[0] - 4.0).abs() < 1e-10, "DC re={}", f.values[0]);
        assert!(f.values[1].abs() < 1e-10, "DC im={}", f.values[1]);
        for i in 1..4 {
            assert!(
                f.values[i * 2].abs() < 1e-10,
                "bin {i} re={}",
                f.values[i * 2]
            );
            assert!(
                f.values[i * 2 + 1].abs() < 1e-10,
                "bin {i} im={}",
                f.values[i * 2 + 1]
            );
        }
    }

    #[test]
    fn fft_ifft_roundtrip() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let a = UFuncArray::new(vec![8], vals.clone(), DType::F64).unwrap();
        let f = a.fft(None).unwrap();
        let recovered = f.ifft().unwrap();
        for (i, &v) in vals.iter().enumerate() {
            assert!(
                (recovered.values[i * 2] - v).abs() < 1e-10,
                "roundtrip [{i}]: got {} expected {}",
                recovered.values[i * 2],
                v
            );
            assert!(
                recovered.values[i * 2 + 1].abs() < 1e-10,
                "imag [{i}] should be ~0: {}",
                recovered.values[i * 2 + 1]
            );
        }
    }

    #[test]
    fn fft_non_power_of_two() {
        // FFT of length 6 (non-power-of-2, uses Bluestein)
        let vals = vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0];
        let a = UFuncArray::new(vec![6], vals.clone(), DType::F64).unwrap();
        let f = a.fft(None).unwrap();
        let recovered = f.ifft().unwrap();
        for (i, &v) in vals.iter().enumerate() {
            assert!(
                (recovered.values[i * 2] - v).abs() < 1e-9,
                "roundtrip [{i}]: got {} expected {}",
                recovered.values[i * 2],
                v
            );
        }
    }

    #[test]
    fn rfft_half_spectrum() {
        let a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let rf = a.rfft(None).unwrap();
        // rfft returns n//2 + 1 = 3 complex coefficients
        assert_eq!(rf.shape, vec![3, 2]);
        // DC component = sum = 10
        assert!((rf.values[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn fftfreq_standard() {
        let f = UFuncArray::fftfreq(8, 1.0);
        assert_eq!(f.shape, vec![8]);
        // Expected: [0, 1/8, 2/8, 3/8, -4/8, -3/8, -2/8, -1/8]
        let expected = [0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125];
        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (f.values[i] - exp).abs() < 1e-10,
                "fftfreq[{i}] = {}, expected {}",
                f.values[i],
                exp
            );
        }
    }

    #[test]
    fn fftshift_centers_zero_frequency() {
        let f = UFuncArray::fftfreq(8, 1.0);
        let shifted = f.fftshift().unwrap();
        // After shift: [-0.5, -0.375, -0.25, -0.125, 0, 0.125, 0.25, 0.375]
        assert!((shifted.values[0] - (-0.5)).abs() < 1e-10);
        assert!((shifted.values[4] - 0.0).abs() < 1e-10);
        // Inverse shift should recover original
        let unshifted = shifted.ifftshift().unwrap();
        for i in 0..8 {
            assert!((unshifted.values[i] - f.values[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn rfftfreq_standard() {
        let f = UFuncArray::rfftfreq(8, 1.0);
        assert_eq!(f.shape, vec![5]); // n//2 + 1 = 5
        let expected = [0.0, 0.125, 0.25, 0.375, 0.5];
        for (i, &exp) in expected.iter().enumerate() {
            assert!((f.values[i] - exp).abs() < 1e-10);
        }
    }

    #[test]
    fn count_nonzero_mixed() {
        let a = UFuncArray::new(vec![5], vec![0.0, 1.0, 0.0, 3.0, -2.0], DType::F64).unwrap();
        assert_eq!(a.count_nonzero(), 3);
        let z = UFuncArray::new(vec![3], vec![0.0, 0.0, 0.0], DType::F64).unwrap();
        assert_eq!(z.count_nonzero(), 0);
    }

    #[test]
    fn packbits_and_unpackbits_roundtrip() {
        // Binary array [1, 0, 1, 1, 0, 0, 1, 0] → packed byte = 0b10110010 = 178
        let bits = vec![1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        let a = UFuncArray::new(vec![8], bits.clone(), DType::U8).unwrap();
        let packed = a.packbits();
        assert_eq!(packed.shape, vec![1]);
        assert!(
            (packed.values[0] - 178.0).abs() < 1e-10,
            "packed={}",
            packed.values[0]
        );

        // Unpack should recover the original bits
        let unpacked = packed.unpackbits();
        assert_eq!(unpacked.shape, vec![8]);
        for (i, &b) in bits.iter().enumerate() {
            assert!(
                (unpacked.values[i] - b).abs() < 1e-10,
                "bit[{i}]={}, expected {}",
                unpacked.values[i],
                b
            );
        }
    }

    #[test]
    fn packbits_partial_byte() {
        // 5 bits → 1 byte (padded with zeros)
        let a = UFuncArray::new(vec![5], vec![1.0, 1.0, 1.0, 0.0, 1.0], DType::U8).unwrap();
        let packed = a.packbits();
        assert_eq!(packed.shape, vec![1]);
        // 11101000 = 232
        assert!(
            (packed.values[0] - 232.0).abs() < 1e-10,
            "packed={}",
            packed.values[0]
        );
    }

    #[test]
    fn mode_returns_most_frequent() {
        let a =
            UFuncArray::new(vec![7], vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0], DType::F64).unwrap();
        let (val, count) = a.mode().unwrap();
        assert!((val - 3.0).abs() < 1e-10);
        assert_eq!(count, 3);
    }

    #[test]
    fn mode_tie_returns_smallest() {
        let a = UFuncArray::new(vec![4], vec![1.0, 1.0, 2.0, 2.0], DType::F64).unwrap();
        let (val, count) = a.mode().unwrap();
        assert!((val - 1.0).abs() < 1e-10);
        assert_eq!(count, 2);
    }

    #[test]
    fn entropy_uniform_distribution() {
        // Uniform distribution over 4 outcomes: entropy = ln(4)
        let a = UFuncArray::new(vec![4], vec![0.25, 0.25, 0.25, 0.25], DType::F64).unwrap();
        let h = a.entropy().unwrap();
        assert!(
            (h - 4.0_f64.ln()).abs() < 1e-10,
            "entropy={h}, expected {}",
            4.0_f64.ln()
        );
    }

    #[test]
    fn entropy_certain_event() {
        // Single outcome with probability 1: entropy = 0
        let a = UFuncArray::new(vec![3], vec![1.0, 0.0, 0.0], DType::F64).unwrap();
        let h = a.entropy().unwrap();
        assert!(
            h.abs() < 1e-10,
            "entropy of certain event should be 0, got {h}"
        );
    }

    #[test]
    fn fromiter_collects_values() {
        let a = UFuncArray::fromiter((0..5).map(|i| i as f64 * 2.0), DType::F64);
        assert_eq!(a.shape, vec![5]);
        assert_eq!(a.values, vec![0.0, 2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn frombuffer_copies_data() {
        let data = [1.0, 2.0, 3.0];
        let a = UFuncArray::frombuffer(&data, DType::F64);
        assert_eq!(a.shape, vec![3]);
        assert_eq!(a.values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn array_equal_and_equiv() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let c = UFuncArray::new(vec![3], vec![1.0, 2.0, 4.0], DType::F64).unwrap();
        assert!(a.array_equal(&b));
        assert!(!a.array_equal(&c));
        assert!(a.array_equiv(&b));
        assert!(!a.array_equiv(&c));
    }

    #[test]
    fn einsum_matmul() {
        // einsum("ij,jk->ik", A, B) = A @ B
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let b = UFuncArray::new(
            vec![3, 2],
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            DType::F64,
        )
        .unwrap();
        let c = UFuncArray::einsum("ij,jk->ik", &[&a, &b]).unwrap();
        assert_eq!(c.shape, vec![2, 2]);
        // Row 0: 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
        // Row 1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
        assert!((c.values[0] - 58.0).abs() < 1e-10);
        assert!((c.values[1] - 64.0).abs() < 1e-10);
        assert!((c.values[2] - 139.0).abs() < 1e-10);
        assert!((c.values[3] - 154.0).abs() < 1e-10);
    }

    #[test]
    fn einsum_dot_product() {
        // einsum("i,i->", a, b) = dot(a, b)
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![4.0, 5.0, 6.0], DType::F64).unwrap();
        let c = UFuncArray::einsum("i,i->", &[&a, &b]).unwrap();
        assert_eq!(c.shape, Vec::<usize>::new());
        assert!((c.values[0] - 32.0).abs() < 1e-10); // 1*4+2*5+3*6=32
    }

    #[test]
    fn einsum_trace() {
        // einsum("ii->", A) = trace(A)
        let a = UFuncArray::new(
            vec![3, 3],
            vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
            DType::F64,
        )
        .unwrap();
        let t = UFuncArray::einsum("ii->", &[&a]).unwrap();
        assert!((t.values[0] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn einsum_sum_axis() {
        // einsum("ij->j", A) = sum along axis 0
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let s = UFuncArray::einsum("ij->j", &[&a]).unwrap();
        assert_eq!(s.shape, vec![3]);
        assert!((s.values[0] - 5.0).abs() < 1e-10); // 1+4
        assert!((s.values[1] - 7.0).abs() < 1e-10); // 2+5
        assert!((s.values[2] - 9.0).abs() < 1e-10); // 3+6
    }

    #[test]
    fn einsum_outer_product() {
        // einsum("i,j->ij", a, b) = outer(a, b)
        let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![3.0, 4.0, 5.0], DType::F64).unwrap();
        let c = UFuncArray::einsum("i,j->ij", &[&a, &b]).unwrap();
        assert_eq!(c.shape, vec![2, 3]);
        assert!((c.values[0] - 3.0).abs() < 1e-10);
        assert!((c.values[1] - 4.0).abs() < 1e-10);
        assert!((c.values[2] - 5.0).abs() < 1e-10);
        assert!((c.values[3] - 6.0).abs() < 1e-10);
        assert!((c.values[4] - 8.0).abs() < 1e-10);
        assert!((c.values[5] - 10.0).abs() < 1e-10);
    }
}
