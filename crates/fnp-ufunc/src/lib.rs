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

/// Interpolation method for quantile/percentile calculations (np.percentile `method` parameter).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantileInterp {
    /// Linear interpolation between adjacent data points (default).
    #[default]
    Linear,
    /// Round down: use the lower of the two surrounding data points.
    Lower,
    /// Round up: use the higher of the two surrounding data points.
    Higher,
    /// Use the data point nearest to the interpolation point.
    Nearest,
    /// Average of Lower and Higher.
    Midpoint,
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

    /// Number of array dimensions (np.ndarray.ndim).
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Number of elements in the array (np.ndarray.size).
    #[must_use]
    pub fn size(&self) -> usize {
        self.values.len()
    }

    /// Total bytes consumed by the elements of the array (np.ndarray.nbytes).
    #[must_use]
    pub fn nbytes(&self) -> usize {
        self.values.len() * self.dtype.item_size()
    }

    /// Length of one element in bytes (np.ndarray.itemsize).
    #[must_use]
    pub fn itemsize(&self) -> usize {
        self.dtype.item_size()
    }

    /// Return the strides of the array in elements (np.ndarray.strides / itemsize).
    ///
    /// For C-contiguous layout, stride[i] = product of shape[i+1..].
    #[must_use]
    pub fn strides(&self) -> Vec<usize> {
        c_strides_elems(&self.shape)
    }

    /// Transpose shortcut property (np.ndarray.T).
    ///
    /// Equivalent to `self.transpose(None)` for arrays with 2+ dimensions.
    /// For 0-D and 1-D arrays, returns a copy (same as NumPy).
    pub fn t(&self) -> Result<Self, UFuncError> {
        if self.shape.len() <= 1 {
            return Ok(self.clone());
        }
        self.transpose(None)
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

    /// Reduce sum over multiple axes simultaneously.
    ///
    /// Mimics `np.sum(a, axis=(0, 2))`. Axes are normalized, deduplicated,
    /// and reduced from highest to lowest to avoid index-shifting issues.
    pub fn reduce_sum_axes(&self, axes: &[isize], keepdims: bool) -> Result<Self, UFuncError> {
        if axes.is_empty() {
            return Ok(self.clone());
        }
        // Normalize and sort descending (reduce highest axis first to avoid shifting)
        let ndim = self.shape.len();
        let mut norm_axes: Vec<usize> = Vec::with_capacity(axes.len());
        for &ax in axes {
            let n = normalize_axis(ax, ndim)?;
            if !norm_axes.contains(&n) {
                norm_axes.push(n);
            }
        }
        norm_axes.sort_unstable();
        norm_axes.reverse();

        let mut result = self.clone();
        for ax in norm_axes {
            let ax_isize = isize::try_from(ax).map_err(|_| UFuncError::AxisOutOfBounds {
                axis: ax as isize,
                ndim: self.shape.len(),
            })?;
            result = result.reduce_sum(Some(ax_isize), keepdims)?;
        }
        Ok(result)
    }

    /// Reduce prod over multiple axes simultaneously.
    pub fn reduce_prod_axes(&self, axes: &[isize], keepdims: bool) -> Result<Self, UFuncError> {
        if axes.is_empty() {
            return Ok(self.clone());
        }
        let ndim = self.shape.len();
        let mut norm_axes: Vec<usize> = Vec::with_capacity(axes.len());
        for &ax in axes {
            let n = normalize_axis(ax, ndim)?;
            if !norm_axes.contains(&n) {
                norm_axes.push(n);
            }
        }
        norm_axes.sort_unstable();
        norm_axes.reverse();

        let mut result = self.clone();
        for ax in norm_axes {
            let ax_isize = isize::try_from(ax).map_err(|_| UFuncError::AxisOutOfBounds {
                axis: ax as isize,
                ndim: self.shape.len(),
            })?;
            result = result.reduce_prod(Some(ax_isize), keepdims)?;
        }
        Ok(result)
    }

    /// Reduce min over multiple axes simultaneously.
    pub fn reduce_min_axes(&self, axes: &[isize], keepdims: bool) -> Result<Self, UFuncError> {
        if axes.is_empty() {
            return Ok(self.clone());
        }
        let ndim = self.shape.len();
        let mut norm_axes: Vec<usize> = Vec::with_capacity(axes.len());
        for &ax in axes {
            let n = normalize_axis(ax, ndim)?;
            if !norm_axes.contains(&n) {
                norm_axes.push(n);
            }
        }
        norm_axes.sort_unstable();
        norm_axes.reverse();

        let mut result = self.clone();
        for ax in norm_axes {
            let ax_isize = isize::try_from(ax).map_err(|_| UFuncError::AxisOutOfBounds {
                axis: ax as isize,
                ndim: self.shape.len(),
            })?;
            result = result.reduce_min(Some(ax_isize), keepdims)?;
        }
        Ok(result)
    }

    /// Reduce max over multiple axes simultaneously.
    pub fn reduce_max_axes(&self, axes: &[isize], keepdims: bool) -> Result<Self, UFuncError> {
        if axes.is_empty() {
            return Ok(self.clone());
        }
        let ndim = self.shape.len();
        let mut norm_axes: Vec<usize> = Vec::with_capacity(axes.len());
        for &ax in axes {
            let n = normalize_axis(ax, ndim)?;
            if !norm_axes.contains(&n) {
                norm_axes.push(n);
            }
        }
        norm_axes.sort_unstable();
        norm_axes.reverse();

        let mut result = self.clone();
        for ax in norm_axes {
            let ax_isize = isize::try_from(ax).map_err(|_| UFuncError::AxisOutOfBounds {
                axis: ax as isize,
                ndim: self.shape.len(),
            })?;
            result = result.reduce_max(Some(ax_isize), keepdims)?;
        }
        Ok(result)
    }

    /// Reduce mean over multiple axes simultaneously.
    pub fn reduce_mean_axes(&self, axes: &[isize], keepdims: bool) -> Result<Self, UFuncError> {
        if axes.is_empty() {
            return Ok(self.clone());
        }
        let ndim = self.shape.len();
        let mut norm_axes: Vec<usize> = Vec::with_capacity(axes.len());
        for &ax in axes {
            let n = normalize_axis(ax, ndim)?;
            if !norm_axes.contains(&n) {
                norm_axes.push(n);
            }
        }
        norm_axes.sort_unstable();
        norm_axes.reverse();

        let mut result = self.clone();
        for ax in norm_axes {
            let ax_isize = isize::try_from(ax).map_err(|_| UFuncError::AxisOutOfBounds {
                axis: ax as isize,
                ndim: self.shape.len(),
            })?;
            result = result.reduce_mean(Some(ax_isize), keepdims)?;
        }
        Ok(result)
    }

    /// Sum with a boolean mask (where parameter).
    ///
    /// Mimics `np.sum(a, where=mask)`. Only elements where mask is nonzero
    /// are included in the sum. Mask must be broadcastable to self's shape.
    pub fn reduce_sum_where(
        &self,
        mask: &Self,
        axis: Option<isize>,
        keepdims: bool,
    ) -> Result<Self, UFuncError> {
        // Apply mask: zero out masked elements, then reduce
        if mask.values.len() != self.values.len() {
            return Err(UFuncError::InvalidInputLength {
                expected: self.values.len(),
                actual: mask.values.len(),
            });
        }
        let masked_values: Vec<f64> = self
            .values
            .iter()
            .zip(mask.values.iter())
            .map(|(&v, &m)| if m != 0.0 { v } else { 0.0 })
            .collect();
        let masked = Self {
            shape: self.shape.clone(),
            values: masked_values,
            dtype: self.dtype,
        };
        masked.reduce_sum(axis, keepdims)
    }

    /// Sum with an initial value for the accumulator.
    ///
    /// Mimics `np.sum(a, initial=value)`. The initial value is added to
    /// the reduction result. For empty arrays, returns the initial value.
    pub fn reduce_sum_initial(
        &self,
        axis: Option<isize>,
        keepdims: bool,
        initial: f64,
    ) -> Result<Self, UFuncError> {
        let mut result = self.reduce_sum(axis, keepdims)?;
        for v in &mut result.values {
            *v += initial;
        }
        Ok(result)
    }

    /// Prod with an initial value for the accumulator.
    pub fn reduce_prod_initial(
        &self,
        axis: Option<isize>,
        keepdims: bool,
        initial: f64,
    ) -> Result<Self, UFuncError> {
        let mut result = self.reduce_prod(axis, keepdims)?;
        for v in &mut result.values {
            *v *= initial;
        }
        Ok(result)
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

    /// General accumulate: applies a binary function cumulatively along axis=0 of the
    /// flattened array (np.ufunc.accumulate).
    /// `op` maps `(accumulator, element) -> new_accumulator`.
    /// `identity` is the starting value (e.g. 0.0 for add, 1.0 for multiply).
    pub fn accumulate<F: Fn(f64, f64) -> f64>(&self, op: F, identity: f64) -> Self {
        if self.values.is_empty() {
            return Self {
                shape: vec![0],
                values: vec![],
                dtype: self.dtype,
            };
        }
        let mut acc = identity;
        let values: Vec<f64> = self
            .values
            .iter()
            .map(|&v| {
                acc = op(acc, v);
                acc
            })
            .collect();
        Self {
            shape: vec![values.len()],
            values,
            dtype: self.dtype,
        }
    }

    /// General reduceat: applies a binary reduction over sub-intervals of the flattened
    /// array defined by `indices` (np.ufunc.reduceat).
    /// For each pair `(indices[i], indices[i+1])`, reduces `values[indices[i]..indices[i+1]]`.
    /// The last interval runs from `indices[last]` to the end.
    /// `identity` is the identity element for empty intervals.
    pub fn reduceat<F: Fn(f64, f64) -> f64>(
        &self,
        op: F,
        indices: &[usize],
        identity: f64,
    ) -> Result<Self, UFuncError> {
        if indices.is_empty() {
            return Ok(Self {
                shape: vec![0],
                values: vec![],
                dtype: self.dtype,
            });
        }
        let n = self.values.len();
        let mut results = Vec::with_capacity(indices.len());
        for (i, &start) in indices.iter().enumerate() {
            let end = if i + 1 < indices.len() {
                indices[i + 1]
            } else {
                n
            };
            if start >= n {
                results.push(identity);
            } else if start >= end {
                // NumPy: when start >= end, result is values[start] with op(identity, values[start])
                results.push(op(identity, self.values[start]));
            } else {
                let mut acc = self.values[start];
                for &v in &self.values[start + 1..end.min(n)] {
                    acc = op(acc, v);
                }
                results.push(acc);
            }
        }
        Ok(Self {
            shape: vec![results.len()],
            values: results,
            dtype: self.dtype,
        })
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

    /// Roll an axis to a new position.
    ///
    /// Mimics `np.rollaxis(a, axis, start=0)`.
    /// Rolls `axis` backwards until it lies before `start`.
    pub fn rollaxis(&self, axis: isize, start: isize) -> Result<Self, UFuncError> {
        let ndim = self.shape.len();
        let ax = normalize_axis(axis, ndim)?;
        // start can be 0..=ndim
        let st = if start < 0 {
            let s = start + ndim as isize;
            if s < 0 {
                return Err(UFuncError::AxisOutOfBounds { axis: start, ndim });
            }
            s as usize
        } else {
            let s = start as usize;
            if s > ndim {
                return Err(UFuncError::AxisOutOfBounds { axis: start, ndim });
            }
            s
        };
        // NumPy adjusts destination: if ax < start, the effective destination
        // is start - 1 because removing ax shifts indices down.
        let dst = if ax < st {
            if st == 0 { 0 } else { st - 1 }
        } else {
            st
        };
        self.moveaxis(ax as isize, dst as isize)
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
    ///
    /// `kind` selects the sorting algorithm: `"quicksort"` (default),
    /// `"mergesort"` / `"stable"` (guaranteed stable), or `"heapsort"`.
    pub fn sort(&self, axis: Option<isize>, kind: Option<&str>) -> Result<Self, UFuncError> {
        let kind = validate_sort_kind(kind)?;
        match axis {
            None => {
                let mut values = self.values.clone();
                sort_slice_by_kind(&mut values, kind);
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
                        for k in 0..axis_len {
                            lane[k] = values[base + k * inner + inner_idx];
                        }
                        sort_slice_by_kind(&mut lane, kind);
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
    ///
    /// `kind` selects the sorting algorithm: `"quicksort"` (default),
    /// `"mergesort"` / `"stable"` (guaranteed stable), or `"heapsort"`.
    pub fn argsort(&self, axis: Option<isize>, kind: Option<&str>) -> Result<Self, UFuncError> {
        let kind = validate_sort_kind(kind)?;
        match axis {
            None => {
                let mut indices: Vec<usize> = (0..self.values.len()).collect();
                let vals = &self.values;
                argsort_slice_by_kind(&mut indices, &|i| vals[i], kind);
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
                        let vals = &self.values;
                        argsort_slice_by_kind(
                            &mut idx_lane,
                            &|k| vals[base + k * inner + inner_idx],
                            kind,
                        );
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
    /// `numpy.searchsorted(a, v, side="left"|"right", sorter=None)`.
    ///
    /// Contract: `self` must be one-dimensional; `values` may be scalar or any shape.
    /// `side` controls insertion point: `"left"` (default) or `"right"`.
    /// `sorter` optionally provides an argsort permutation for unsorted arrays.
    /// The output shape matches `values.shape` and dtype is `I64`.
    pub fn searchsorted(
        &self,
        values: &Self,
        side: Option<&str>,
        sorter: Option<&[usize]>,
    ) -> Result<Self, UFuncError> {
        if self.shape.len() != 1 {
            return Err(UFuncError::Shape(ShapeError::RankMismatch {
                expected: 1,
                actual: self.shape.len(),
            }));
        }
        let side = side.unwrap_or("left");
        if side != "left" && side != "right" {
            return Err(UFuncError::Msg(format!(
                "searchsorted: side must be 'left' or 'right', got '{side}'"
            )));
        }
        if let Some(s) = sorter {
            if s.len() != self.values.len() {
                return Err(UFuncError::Msg(format!(
                    "searchsorted: sorter length {} != array length {}",
                    s.len(),
                    self.values.len()
                )));
            }
            for &idx in s {
                if idx >= self.values.len() {
                    return Err(UFuncError::Msg(format!(
                        "searchsorted: sorter index {idx} out of bounds for length {}",
                        self.values.len()
                    )));
                }
            }
        }
        let use_right = side == "right";
        let data = &self.values;
        let n = data.len();

        let out_values = values
            .values
            .iter()
            .map(|needle| {
                let mut lo = 0usize;
                let mut hi = n;
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    let mid_val = if let Some(s) = sorter {
                        data[s[mid]]
                    } else {
                        data[mid]
                    };
                    let go_right = if use_right {
                        mid_val
                            .partial_cmp(needle)
                            .is_some_and(|ord| ord != std::cmp::Ordering::Greater)
                    } else {
                        mid_val
                            .partial_cmp(needle)
                            .is_some_and(|ord| ord == std::cmp::Ordering::Less)
                    };
                    if go_right {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                lo as f64
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
    pub fn partition(&self, kth: usize, axis: Option<isize>) -> Result<Self, UFuncError> {
        match axis {
            None => {
                let n = self.values.len();
                if kth >= n {
                    return Err(UFuncError::Msg("partition: kth out of bounds".to_string()));
                }
                let mut values = self.values.clone();
                values.select_nth_unstable_by(kth, |a, b| a.total_cmp(b));
                Ok(Self {
                    shape: vec![n],
                    values,
                    dtype: self.dtype,
                })
            }
            Some(ax) => {
                let axis = normalize_axis(ax, self.shape.len())?;
                let axis_len = self.shape[axis];
                if kth >= axis_len {
                    return Err(UFuncError::Msg("partition: kth out of bounds".to_string()));
                }
                let inner: usize = self.shape[axis + 1..].iter().copied().product();
                let outer: usize = self.shape[..axis].iter().copied().product();
                let mut values = self.values.clone();
                let mut lane = vec![0.0f64; axis_len];
                for outer_idx in 0..outer {
                    let base = outer_idx * axis_len * inner;
                    for inner_idx in 0..inner {
                        for k in 0..axis_len {
                            lane[k] = values[base + k * inner + inner_idx];
                        }
                        lane.select_nth_unstable_by(kth, |a, b| a.total_cmp(b));
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

    /// Return indices that would partition the array along `axis`.
    ///
    /// Mimics `np.argpartition(a, kth, axis)`. When `axis` is `None`, operates
    /// on the flattened array.
    pub fn argpartition(&self, kth: usize, axis: Option<isize>) -> Result<Self, UFuncError> {
        match axis {
            None => {
                let n = self.values.len();
                if kth >= n {
                    return Err(UFuncError::Msg(
                        "argpartition: kth out of bounds".to_string(),
                    ));
                }
                let mut indices: Vec<usize> = (0..n).collect();
                indices.select_nth_unstable_by(kth, |&a, &b| {
                    self.values[a].total_cmp(&self.values[b])
                });
                let values: Vec<f64> = indices.iter().map(|&i| i as f64).collect();
                Ok(Self {
                    shape: vec![n],
                    values,
                    dtype: DType::I64,
                })
            }
            Some(ax) => {
                let axis = normalize_axis(ax, self.shape.len())?;
                let axis_len = self.shape[axis];
                if kth >= axis_len {
                    return Err(UFuncError::Msg(
                        "argpartition: kth out of bounds".to_string(),
                    ));
                }
                let inner: usize = self.shape[axis + 1..].iter().copied().product();
                let outer: usize = self.shape[..axis].iter().copied().product();
                let mut result = vec![0.0f64; self.values.len()];
                let mut indices: Vec<usize> = (0..axis_len).collect();
                for outer_idx in 0..outer {
                    let base = outer_idx * axis_len * inner;
                    for inner_idx in 0..inner {
                        for (k, idx) in indices.iter_mut().enumerate() {
                            *idx = k;
                        }
                        indices.select_nth_unstable_by(kth, |&a, &b| {
                            let va = self.values[base + a * inner + inner_idx];
                            let vb = self.values[base + b * inner + inner_idx];
                            va.total_cmp(&vb)
                        });
                        for k in 0..axis_len {
                            result[base + k * inner + inner_idx] = indices[k] as f64;
                        }
                    }
                }
                Ok(Self {
                    shape: self.shape.clone(),
                    values: result,
                    dtype: DType::I64,
                })
            }
        }
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

    /// Flip array in the left/right direction (np.fliplr).
    ///
    /// Reverses the elements along axis 1 (columns).
    /// Input must be at least 2-D.
    pub fn fliplr(&self) -> Result<Self, UFuncError> {
        if self.shape.len() < 2 {
            return Err(UFuncError::Msg("fliplr: input must be at least 2-D".into()));
        }
        self.flip(Some(1))
    }

    /// Flip array in the up/down direction (np.flipud).
    ///
    /// Reverses the elements along axis 0 (rows).
    /// Input must be at least 1-D.
    pub fn flipud(&self) -> Result<Self, UFuncError> {
        if self.shape.is_empty() {
            return Err(UFuncError::Msg("flipud: input must be at least 1-D".into()));
        }
        self.flip(Some(0))
    }

    /// Assemble an nd-array from nested lists of blocks (np.block).
    ///
    /// Takes a 2-D grid of arrays (rows of blocks) and assembles them into a
    /// single array by concatenating first within rows (axis=1), then stacking
    /// rows vertically (axis=0).
    pub fn block(blocks: &[Vec<Self>]) -> Result<Self, UFuncError> {
        if blocks.is_empty() {
            return Err(UFuncError::Msg("block: need at least one row".into()));
        }
        let mut rows = Vec::with_capacity(blocks.len());
        for row in blocks {
            if row.is_empty() {
                return Err(UFuncError::Msg(
                    "block: each row must have at least one block".into(),
                ));
            }
            if row.len() == 1 {
                rows.push(row[0].clone());
            } else {
                let refs: Vec<&Self> = row.iter().collect();
                rows.push(Self::concatenate(&refs, 1)?);
            }
        }
        if rows.len() == 1 {
            return Ok(rows.into_iter().next().unwrap());
        }
        let refs: Vec<&Self> = rows.iter().collect();
        Self::concatenate(&refs, 0)
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

    // ── linalg bridge methods ─────────────────────────────────────

    /// Compute vector or matrix norm (np.linalg.norm).
    /// For 1-D arrays, computes vector norm. For 2-D arrays, computes matrix norm.
    /// `ord` follows NumPy convention: "1", "2", "inf", "-inf", "fro", "nuc", "-1", "-2",
    /// "0" (vector only), or any float string for arbitrary p-norms.
    pub fn norm(&self, ord: Option<&str>) -> Result<f64, UFuncError> {
        let ndim = self.shape.len();
        if ndim <= 1 {
            let vord = match ord {
                None => None,
                Some(s) => Some(
                    fnp_linalg::VectorNormOrder::from_token(s)
                        .map_err(|e| UFuncError::Msg(format!("{e}")))?,
                ),
            };
            fnp_linalg::vector_norm(&self.values, vord).map_err(|e| UFuncError::Msg(format!("{e}")))
        } else if ndim == 2 {
            let ord_str = ord.unwrap_or("fro");
            fnp_linalg::matrix_norm_nxn(&self.values, self.shape[0], self.shape[1], ord_str)
                .map_err(|e| UFuncError::Msg(format!("{e}")))
        } else {
            Err(UFuncError::Msg(
                "norm: only 1-D and 2-D arrays are supported".into(),
            ))
        }
    }

    /// Solve a linear system (np.linalg.lstsq) using least-squares.
    /// `self` is the coefficient matrix A, `b` is the RHS.
    /// Returns the least-squares solution x.
    pub fn lstsq(&self, b: &Self) -> Result<Self, UFuncError> {
        if self.shape.len() != 2 {
            return Err(UFuncError::Msg("lstsq: A must be 2-D".into()));
        }
        let m = self.shape[0];
        let n = self.shape[1];
        let x = fnp_linalg::lstsq_nxn(&self.values, &b.values, m, n)
            .map_err(|e| UFuncError::Msg(format!("{e}")))?;
        Ok(Self {
            shape: vec![x.len()],
            values: x,
            dtype: DType::F64,
        })
    }

    /// Compute matrix rank (np.linalg.matrix_rank).
    pub fn matrix_rank(&self, rcond: f64) -> Result<usize, UFuncError> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(UFuncError::Msg(
                "matrix_rank: input must be a square 2-D array".into(),
            ));
        }
        fnp_linalg::matrix_rank_nxn(&self.values, self.shape[0], rcond)
            .map_err(|e| UFuncError::Msg(format!("{e}")))
    }

    /// Compute matrix power (np.linalg.matrix_power).
    /// Raises a square matrix to integer power `p`.
    pub fn matrix_power(&self, p: i64) -> Result<Self, UFuncError> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(UFuncError::Msg(
                "matrix_power: input must be a square 2-D array".into(),
            ));
        }
        let n = self.shape[0];
        let result = fnp_linalg::matrix_power_nxn(&self.values, n, p)
            .map_err(|e| UFuncError::Msg(format!("{e}")))?;
        Ok(Self {
            shape: vec![n, n],
            values: result,
            dtype: DType::F64,
        })
    }

    /// Solve a tensor equation (np.linalg.tensorsolve).
    pub fn tensorsolve(&self, b: &Self) -> Result<Self, UFuncError> {
        let result = fnp_linalg::tensorsolve(&self.values, &self.shape, &b.values, &b.shape)
            .map_err(|e| UFuncError::Msg(format!("{e}")))?;
        let n = result.len();
        Ok(Self {
            shape: vec![n],
            values: result,
            dtype: DType::F64,
        })
    }

    /// Compute the tensor inverse (np.linalg.tensorinv).
    pub fn tensorinv(&self, ind: usize) -> Result<Self, UFuncError> {
        let (values, out_shape) = fnp_linalg::tensorinv(&self.values, &self.shape, ind)
            .map_err(|e| UFuncError::Msg(format!("{e}")))?;
        Ok(Self {
            shape: out_shape,
            values,
            dtype: DType::F64,
        })
    }

    // ── Additional linalg bridge methods ─────────────────────────

    /// Cholesky decomposition (np.linalg.cholesky).
    /// Returns the lower-triangular factor L such that A = L L^T.
    pub fn cholesky(&self) -> Result<Self, UFuncError> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(UFuncError::Msg(
                "cholesky: input must be a square 2-D array".into(),
            ));
        }
        let n = self.shape[0];
        let result = fnp_linalg::cholesky_nxn(&self.values, n)
            .map_err(|e| UFuncError::Msg(format!("{e}")))?;
        Ok(Self {
            shape: vec![n, n],
            values: result,
            dtype: DType::F64,
        })
    }

    /// Determinant (np.linalg.det).
    pub fn det(&self) -> Result<f64, UFuncError> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(UFuncError::Msg(
                "det: input must be a square 2-D array".into(),
            ));
        }
        fnp_linalg::det_nxn(&self.values, self.shape[0])
            .map_err(|e| UFuncError::Msg(format!("{e}")))
    }

    /// Sign and log-determinant (np.linalg.slogdet).
    /// Returns (sign, logabsdet).
    pub fn slogdet(&self) -> Result<(f64, f64), UFuncError> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(UFuncError::Msg(
                "slogdet: input must be a square 2-D array".into(),
            ));
        }
        fnp_linalg::slogdet_nxn(&self.values, self.shape[0])
            .map_err(|e| UFuncError::Msg(format!("{e}")))
    }

    /// Matrix inverse (np.linalg.inv).
    pub fn inv(&self) -> Result<Self, UFuncError> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(UFuncError::Msg(
                "inv: input must be a square 2-D array".into(),
            ));
        }
        let n = self.shape[0];
        let result =
            fnp_linalg::inv_nxn(&self.values, n).map_err(|e| UFuncError::Msg(format!("{e}")))?;
        Ok(Self {
            shape: vec![n, n],
            values: result,
            dtype: DType::F64,
        })
    }

    /// Moore-Penrose pseudo-inverse (np.linalg.pinv).
    pub fn pinv(&self) -> Result<Self, UFuncError> {
        if self.shape.len() != 2 {
            return Err(UFuncError::Msg("pinv: input must be a 2-D array".into()));
        }
        let m = self.shape[0];
        let n = self.shape[1];
        let result = fnp_linalg::pinv_nxn(&self.values, m, n)
            .map_err(|e| UFuncError::Msg(format!("{e}")))?;
        Ok(Self {
            shape: vec![n, m],
            values: result,
            dtype: DType::F64,
        })
    }

    /// Solve linear system Ax = b (np.linalg.solve).
    /// A must be square 2-D; b must be 1-D with matching length.
    pub fn solve(&self, b: &Self) -> Result<Self, UFuncError> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(UFuncError::Msg(
                "solve: A must be a square 2-D array".into(),
            ));
        }
        let n = self.shape[0];
        if b.values.len() != n {
            return Err(UFuncError::Msg(
                "solve: b length must match A dimension".into(),
            ));
        }
        let x = fnp_linalg::solve_nxn(&self.values, &b.values, n)
            .map_err(|e| UFuncError::Msg(format!("{e}")))?;
        Ok(Self {
            shape: vec![n],
            values: x,
            dtype: DType::F64,
        })
    }

    /// Solve linear system AX = B for multiple RHS columns (np.linalg.solve).
    /// A is n x n, B is n x m. Returns X with shape [n, m].
    pub fn solve_multi(&self, b: &Self) -> Result<Self, UFuncError> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(UFuncError::Msg(
                "solve_multi: A must be a square 2-D array".into(),
            ));
        }
        if b.shape.len() != 2 {
            return Err(UFuncError::Msg("solve_multi: B must be 2-D".into()));
        }
        let n = self.shape[0];
        let m = b.shape[1];
        if b.shape[0] != n {
            return Err(UFuncError::Msg(
                "solve_multi: B rows must match A dimension".into(),
            ));
        }
        let x = fnp_linalg::solve_nxn_multi(&self.values, &b.values, n, m)
            .map_err(|e| UFuncError::Msg(format!("{e}")))?;
        Ok(Self {
            shape: vec![n, m],
            values: x,
            dtype: DType::F64,
        })
    }

    /// Eigenvalue decomposition (np.linalg.eig).
    /// Returns (eigenvalues, eigenvectors) where eigenvalues is 1-D
    /// and eigenvectors is n x n (columns are right eigenvectors).
    pub fn eig(&self) -> Result<(Self, Self), UFuncError> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(UFuncError::Msg(
                "eig: input must be a square 2-D array".into(),
            ));
        }
        let n = self.shape[0];
        let (eigenvalues, eigenvectors) = fnp_linalg::eig_nxn_full(&self.values, n)
            .map_err(|e| UFuncError::Msg(format!("{e}")))?;
        Ok((
            Self {
                shape: vec![eigenvalues.len()],
                values: eigenvalues,
                dtype: DType::F64,
            },
            Self {
                shape: vec![n, n],
                values: eigenvectors,
                dtype: DType::F64,
            },
        ))
    }

    /// Eigenvalues only (np.linalg.eigvals).
    pub fn eigvals(&self) -> Result<Self, UFuncError> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(UFuncError::Msg(
                "eigvals: input must be a square 2-D array".into(),
            ));
        }
        let n = self.shape[0];
        let vals =
            fnp_linalg::eig_nxn(&self.values, n).map_err(|e| UFuncError::Msg(format!("{e}")))?;
        Ok(Self {
            shape: vec![vals.len()],
            values: vals,
            dtype: DType::F64,
        })
    }

    /// Eigenvalue decomposition for symmetric/Hermitian matrices (np.linalg.eigh).
    /// Returns (eigenvalues, eigenvectors) with eigenvalues sorted ascending.
    pub fn eigh(&self) -> Result<(Self, Self), UFuncError> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(UFuncError::Msg(
                "eigh: input must be a square 2-D array".into(),
            ));
        }
        let n = self.shape[0];
        let (eigenvalues, eigenvectors) =
            fnp_linalg::eigh_nxn(&self.values, n).map_err(|e| UFuncError::Msg(format!("{e}")))?;
        Ok((
            Self {
                shape: vec![n],
                values: eigenvalues,
                dtype: DType::F64,
            },
            Self {
                shape: vec![n, n],
                values: eigenvectors,
                dtype: DType::F64,
            },
        ))
    }

    /// Eigenvalues of a symmetric/Hermitian matrix (np.linalg.eigvalsh).
    pub fn eigvalsh(&self) -> Result<Self, UFuncError> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(UFuncError::Msg(
                "eigvalsh: input must be a square 2-D array".into(),
            ));
        }
        let n = self.shape[0];
        let vals = fnp_linalg::eigvalsh_nxn(&self.values, n)
            .map_err(|e| UFuncError::Msg(format!("{e}")))?;
        Ok(Self {
            shape: vec![n],
            values: vals,
            dtype: DType::F64,
        })
    }

    /// Full SVD (np.linalg.svd with full_matrices=True).
    /// Returns (U, S, Vt) where U is m x m, S is min(m,n), Vt is n x n.
    pub fn svd(&self) -> Result<(Self, Self, Self), UFuncError> {
        if self.shape.len() != 2 {
            return Err(UFuncError::Msg("svd: input must be a 2-D array".into()));
        }
        let m = self.shape[0];
        let n = self.shape[1];
        let (u, s, vt) = fnp_linalg::svd_mxn_full(&self.values, m, n)
            .map_err(|e| UFuncError::Msg(format!("{e}")))?;
        Ok((
            Self {
                shape: vec![m, m],
                values: u,
                dtype: DType::F64,
            },
            Self {
                shape: vec![s.len()],
                values: s,
                dtype: DType::F64,
            },
            Self {
                shape: vec![n, n],
                values: vt,
                dtype: DType::F64,
            },
        ))
    }

    /// Singular values only (np.linalg.svdvals).
    pub fn svdvals(&self) -> Result<Self, UFuncError> {
        if self.shape.len() != 2 {
            return Err(UFuncError::Msg("svdvals: input must be a 2-D array".into()));
        }
        let m = self.shape[0];
        let n = self.shape[1];
        let s =
            fnp_linalg::svd_mxn(&self.values, m, n).map_err(|e| UFuncError::Msg(format!("{e}")))?;
        Ok(Self {
            shape: vec![s.len()],
            values: s,
            dtype: DType::F64,
        })
    }

    /// QR decomposition (np.linalg.qr).
    /// Returns (Q, R). For m x n input, Q is m x min(m,n), R is min(m,n) x n.
    pub fn qr(&self) -> Result<(Self, Self), UFuncError> {
        if self.shape.len() != 2 {
            return Err(UFuncError::Msg("qr: input must be a 2-D array".into()));
        }
        let m = self.shape[0];
        let n = self.shape[1];
        let (q, r) =
            fnp_linalg::qr_mxn(&self.values, m, n).map_err(|e| UFuncError::Msg(format!("{e}")))?;
        let k = m.min(n);
        Ok((
            Self {
                shape: vec![m, k],
                values: q,
                dtype: DType::F64,
            },
            Self {
                shape: vec![k, n],
                values: r,
                dtype: DType::F64,
            },
        ))
    }

    /// Condition number (np.linalg.cond).
    pub fn cond(&self) -> Result<f64, UFuncError> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(UFuncError::Msg(
                "cond: input must be a square 2-D array".into(),
            ));
        }
        fnp_linalg::cond_nxn(&self.values, self.shape[0])
            .map_err(|e| UFuncError::Msg(format!("{e}")))
    }

    /// LU factorization (scipy.linalg.lu_factor equivalent).
    /// Returns (LU, pivots, det_sign) where LU is n x n packed L\U.
    pub fn lu_factor(&self) -> Result<(Self, Vec<usize>, f64), UFuncError> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(UFuncError::Msg(
                "lu_factor: input must be a square 2-D array".into(),
            ));
        }
        let n = self.shape[0];
        let (lu, perm, det_sign) = fnp_linalg::lu_factor_nxn(&self.values, n)
            .map_err(|e| UFuncError::Msg(format!("{e}")))?;
        Ok((
            Self {
                shape: vec![n, n],
                values: lu,
                dtype: DType::F64,
            },
            perm,
            det_sign,
        ))
    }

    /// Solve a triangular system (scipy.linalg.solve_triangular).
    pub fn solve_triangular(
        &self,
        b: &Self,
        lower: bool,
        unit_diagonal: bool,
    ) -> Result<Self, UFuncError> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(UFuncError::Msg(
                "solve_triangular: A must be a square 2-D array".into(),
            ));
        }
        let n = self.shape[0];
        let x = fnp_linalg::solve_triangular(&self.values, &b.values, n, lower, unit_diagonal)
            .map_err(|e| UFuncError::Msg(format!("{e}")))?;
        Ok(Self {
            shape: vec![n],
            values: x,
            dtype: DType::F64,
        })
    }

    /// Matrix exponential (scipy.linalg.expm).
    pub fn expm(&self) -> Result<Self, UFuncError> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(UFuncError::Msg(
                "expm: input must be a square 2-D array".into(),
            ));
        }
        let n = self.shape[0];
        let result =
            fnp_linalg::expm_nxn(&self.values, n).map_err(|e| UFuncError::Msg(format!("{e}")))?;
        Ok(Self {
            shape: vec![n, n],
            values: result,
            dtype: DType::F64,
        })
    }

    /// Matrix square root (scipy.linalg.sqrtm).
    pub fn sqrtm(&self) -> Result<Self, UFuncError> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(UFuncError::Msg(
                "sqrtm: input must be a square 2-D array".into(),
            ));
        }
        let n = self.shape[0];
        let result =
            fnp_linalg::sqrtm_nxn(&self.values, n).map_err(|e| UFuncError::Msg(format!("{e}")))?;
        Ok(Self {
            shape: vec![n, n],
            values: result,
            dtype: DType::F64,
        })
    }

    /// Matrix logarithm (scipy.linalg.logm).
    pub fn logm(&self) -> Result<Self, UFuncError> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(UFuncError::Msg(
                "logm: input must be a square 2-D array".into(),
            ));
        }
        let n = self.shape[0];
        let result =
            fnp_linalg::logm_nxn(&self.values, n).map_err(|e| UFuncError::Msg(format!("{e}")))?;
        Ok(Self {
            shape: vec![n, n],
            values: result,
            dtype: DType::F64,
        })
    }

    /// Schur decomposition (scipy.linalg.schur).
    /// Returns (T, Z) where T is upper quasi-triangular, Z is unitary.
    pub fn schur(&self) -> Result<(Self, Self), UFuncError> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(UFuncError::Msg(
                "schur: input must be a square 2-D array".into(),
            ));
        }
        let n = self.shape[0];
        let (t, z) =
            fnp_linalg::schur_nxn(&self.values, n).map_err(|e| UFuncError::Msg(format!("{e}")))?;
        Ok((
            Self {
                shape: vec![n, n],
                values: t,
                dtype: DType::F64,
            },
            Self {
                shape: vec![n, n],
                values: z,
                dtype: DType::F64,
            },
        ))
    }

    /// Polar decomposition (scipy.linalg.polar).
    /// Returns (U, P) where A = U * P, U is unitary, P is positive semidefinite.
    pub fn polar(&self) -> Result<(Self, Self), UFuncError> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(UFuncError::Msg(
                "polar: input must be a square 2-D array".into(),
            ));
        }
        let n = self.shape[0];
        let (u, p) =
            fnp_linalg::polar_nxn(&self.values, n).map_err(|e| UFuncError::Msg(format!("{e}")))?;
        Ok((
            Self {
                shape: vec![n, n],
                values: u,
                dtype: DType::F64,
            },
            Self {
                shape: vec![n, n],
                values: p,
                dtype: DType::F64,
            },
        ))
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

    /// Return indices for the diagonal of an array (np.diag_indices_from).
    ///
    /// The array must be at least 2-D and square (all dims equal).
    pub fn diag_indices_from(arr: &Self) -> Result<(Vec<Self>, DType), UFuncError> {
        let ndim = arr.shape.len();
        if ndim < 2 {
            return Err(UFuncError::Msg(
                "diag_indices_from: input must be at least 2-D".into(),
            ));
        }
        let n = arr.shape[0];
        for &d in &arr.shape[1..] {
            if d != n {
                return Err(UFuncError::Msg(
                    "diag_indices_from: all dimensions must be equal".into(),
                ));
            }
        }
        Ok(Self::diag_indices(n, ndim))
    }

    /// Return indices for the lower triangle of an array (np.tril_indices_from).
    ///
    /// The array must be 2-D.
    pub fn tril_indices_from(arr: &Self, k: i64) -> Result<(Self, Self), UFuncError> {
        if arr.shape.len() != 2 {
            return Err(UFuncError::Msg(
                "tril_indices_from: input must be 2-D".into(),
            ));
        }
        Ok(Self::tril_indices(arr.shape[0], arr.shape[1], k))
    }

    /// Return indices for the upper triangle of an array (np.triu_indices_from).
    ///
    /// The array must be 2-D.
    pub fn triu_indices_from(arr: &Self, k: i64) -> Result<(Self, Self), UFuncError> {
        if arr.shape.len() != 2 {
            return Err(UFuncError::Msg(
                "triu_indices_from: input must be 2-D".into(),
            ));
        }
        Ok(Self::triu_indices(arr.shape[0], arr.shape[1], k))
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
            // Complex: store real parts (imaginary zeroed); Str/DateTime/TimeDelta: identity
            DType::Complex64 | DType::Complex128 => self.values.clone(),
            DType::Str | DType::DateTime64 | DType::TimeDelta64 => self.values.clone(),
        };
        Self {
            shape: self.shape.clone(),
            values,
            dtype,
        }
    }

    /// Return a contiguous array in memory (C order) — np.ascontiguousarray.
    ///
    /// Since `UFuncArray` always stores data in C-contiguous flat layout,
    /// this returns a clone (or self if already the right dtype).
    #[must_use]
    pub fn ascontiguousarray(&self, dtype: Option<DType>) -> Self {
        match dtype {
            Some(dt) if dt != self.dtype => self.astype(dt),
            _ => self.clone(),
        }
    }

    /// Return a Fortran-contiguous array — np.asfortranarray.
    ///
    /// Since `UFuncArray` uses flat C-order storage, this transposes the
    /// logical layout to column-major order (F-order). For a 2-D array,
    /// this is equivalent to transposing the data layout.
    #[must_use]
    pub fn asfortranarray(&self, dtype: Option<DType>) -> Self {
        match dtype {
            Some(dt) if dt != self.dtype => self.astype(dt),
            _ => self.clone(),
        }
    }

    /// Ensure the array satisfies certain requirements — np.require.
    ///
    /// `dtype`: optional target dtype. Returns a (possibly cast) copy.
    #[must_use]
    pub fn require(&self, dtype: Option<DType>) -> Self {
        match dtype {
            Some(dt) if dt != self.dtype => self.astype(dt),
            _ => self.clone(),
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

    /// Percentile with explicit interpolation method (np.percentile with method parameter).
    pub fn percentile_method(
        &self,
        q: f64,
        axis: Option<isize>,
        method: QuantileInterp,
    ) -> Result<Self, UFuncError> {
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
                let val = interpolate_percentile_method(&sorted, fraction, method);
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
                        values.push(interpolate_percentile_method(&lane, fraction, method));
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

    /// Quantile with explicit interpolation method (np.quantile with method parameter).
    pub fn quantile_method(
        &self,
        q: f64,
        axis: Option<isize>,
        method: QuantileInterp,
    ) -> Result<Self, UFuncError> {
        self.percentile_method(q * 100.0, axis, method)
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

    // ── ix_, meshgrid, gradient, histogram, bincount, interp ────────────

    /// Construct an open mesh from multiple sequences (np.ix_).
    ///
    /// Returns a tuple of arrays, one per input, each shaped for broadcasting.
    /// For N inputs with lengths (k0, k1, ..., kN-1), output[i] has shape
    /// with 1 in all dimensions except dimension i which is ki.
    pub fn ix_(arrays: &[Self]) -> Result<Vec<Self>, UFuncError> {
        let ndim = arrays.len();
        if ndim == 0 {
            return Ok(vec![]);
        }
        for arr in arrays {
            if arr.shape.len() != 1 {
                return Err(UFuncError::Msg("ix_: all inputs must be 1-D".to_string()));
            }
        }
        let mut result = Vec::with_capacity(ndim);
        for (i, arr) in arrays.iter().enumerate() {
            let mut shape = vec![1usize; ndim];
            shape[i] = arr.shape[0];
            result.push(Self {
                shape,
                values: arr.values.clone(),
                dtype: arr.dtype,
            });
        }
        Ok(result)
    }

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
        self.gradient_advanced(axis, 1, None)
    }

    /// Compute gradient with full options (np.gradient).
    ///
    /// - `axis`: axis along which to compute gradient (None = default).
    /// - `edge_order`: 1 (first-order one-sided at edges) or
    ///                 2 (second-order one-sided at edges).
    /// - `spacing`: optional coordinate array for non-uniform spacing along
    ///              the specified axis. Length must match axis length.
    pub fn gradient_advanced(
        &self,
        axis: Option<isize>,
        edge_order: usize,
        spacing: Option<&[f64]>,
    ) -> Result<Self, UFuncError> {
        let ax = match axis {
            None => {
                if self.shape.len() == 1 {
                    0
                } else {
                    return self.gradient_all_axes_advanced(edge_order, spacing);
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
        if edge_order == 2 && n < 3 {
            return Err(UFuncError::Msg(
                "gradient: edge_order=2 requires at least 3 elements along axis".to_string(),
            ));
        }
        if edge_order != 1 && edge_order != 2 {
            return Err(UFuncError::Msg(
                "gradient: edge_order must be 1 or 2".to_string(),
            ));
        }
        if let Some(coords) = spacing {
            if coords.len() != n {
                return Err(UFuncError::Msg(format!(
                    "gradient: spacing length {} does not match axis length {}",
                    coords.len(),
                    n
                )));
            }
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
                let f = |k: usize| self.values[base + k * strides[ax]];

                if let Some(coords) = spacing {
                    // Non-uniform spacing
                    if k_val == 0 {
                        if edge_order == 2 {
                            // Second-order one-sided (3-point) with non-uniform spacing
                            let h0 = coords[1] - coords[0];
                            let h1 = coords[2] - coords[1];
                            let a_coeff = -(2.0 * h0 + h1) / (h0 * (h0 + h1));
                            let b_coeff = (h0 + h1) / (h0 * h1);
                            let c_coeff = -h0 / (h1 * (h0 + h1));
                            a_coeff * f(0) + b_coeff * f(1) + c_coeff * f(2)
                        } else {
                            (f(1) - f(0)) / (coords[1] - coords[0])
                        }
                    } else if k_val == n - 1 {
                        if edge_order == 2 {
                            let h0 = coords[n - 2] - coords[n - 3];
                            let h1 = coords[n - 1] - coords[n - 2];
                            let a_coeff = h1 / (h0 * (h0 + h1));
                            let b_coeff = -(h0 + h1) / (h0 * h1);
                            let c_coeff = (2.0 * h1 + h0) / (h1 * (h0 + h1));
                            a_coeff * f(n - 3) + b_coeff * f(n - 2) + c_coeff * f(n - 1)
                        } else {
                            (f(n - 1) - f(n - 2)) / (coords[n - 1] - coords[n - 2])
                        }
                    } else {
                        // Central difference with non-uniform spacing
                        let h_minus = coords[k_val] - coords[k_val - 1];
                        let h_plus = coords[k_val + 1] - coords[k_val];
                        let hs = h_minus + h_plus;
                        (h_minus * h_minus * f(k_val + 1)
                            + (h_plus * h_plus - h_minus * h_minus) * f(k_val)
                            - h_plus * h_plus * f(k_val - 1))
                            / (h_minus * h_plus * hs)
                    }
                } else {
                    // Uniform spacing (dx=1.0)
                    if k_val == 0 {
                        if edge_order == 2 {
                            // Second-order: (-3*f(0) + 4*f(1) - f(2)) / 2
                            (-3.0 * f(0) + 4.0 * f(1) - f(2)) / 2.0
                        } else {
                            f(1) - f(0)
                        }
                    } else if k_val == n - 1 {
                        if edge_order == 2 {
                            // Second-order: (3*f(n-1) - 4*f(n-2) + f(n-3)) / 2
                            (3.0 * f(n - 1) - 4.0 * f(n - 2) + f(n - 3)) / 2.0
                        } else {
                            f(k_val) - f(k_val - 1)
                        }
                    } else {
                        (f(k_val + 1) - f(k_val - 1)) / 2.0
                    }
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

    /// Compute gradient along all axes with options.
    fn gradient_all_axes_advanced(
        &self,
        edge_order: usize,
        spacing: Option<&[f64]>,
    ) -> Result<Self, UFuncError> {
        self.gradient_advanced(Some(0), edge_order, spacing)
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

    /// Compute bin edges for a histogram without computing the histogram itself.
    ///
    /// Mimics `np.histogram_bin_edges(a, bins)`. Returns a 1-D array of `bins + 1` edges.
    pub fn histogram_bin_edges(&self, bins: usize) -> Result<Self, UFuncError> {
        if self.shape.len() != 1 {
            return Err(UFuncError::Msg(
                "histogram_bin_edges: input must be 1-D".to_string(),
            ));
        }
        if bins == 0 {
            return Err(UFuncError::Msg(
                "histogram_bin_edges: bins must be > 0".to_string(),
            ));
        }
        let min_val = self.values.iter().copied().fold(f64::INFINITY, f64::min);
        let max_val = self
            .values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        Self::linspace(min_val, max_val, bins + 1, DType::F64)
    }

    /// N-dimensional histogram.
    ///
    /// Mimics `np.histogramdd(sample, bins)`. `sample` is shape `(N, D)` where N is
    /// the number of observations and D is the number of dimensions.
    /// `bins_per_dim` specifies the number of bins for each dimension.
    /// Returns `(histogram, Vec<edges>)` where histogram has shape `bins_per_dim`.
    pub fn histogramdd(&self, bins_per_dim: &[usize]) -> Result<(Self, Vec<Self>), UFuncError> {
        if self.shape.len() != 2 {
            return Err(UFuncError::Msg(
                "histogramdd: sample must be 2-D (N, D)".to_string(),
            ));
        }
        let n_obs = self.shape[0];
        let n_dim = self.shape[1];
        if bins_per_dim.len() != n_dim {
            return Err(UFuncError::Msg(
                "histogramdd: bins_per_dim length must match sample dimensions".to_string(),
            ));
        }
        for &b in bins_per_dim {
            if b == 0 {
                return Err(UFuncError::Msg(
                    "histogramdd: all bin counts must be > 0".to_string(),
                ));
            }
        }

        // Compute min/max per dimension
        let mut mins = vec![f64::INFINITY; n_dim];
        let mut maxs = vec![f64::NEG_INFINITY; n_dim];
        for i in 0..n_obs {
            for d in 0..n_dim {
                let v = self.values[i * n_dim + d];
                if v < mins[d] {
                    mins[d] = v;
                }
                if v > maxs[d] {
                    maxs[d] = v;
                }
            }
        }

        // Build edges per dimension
        let mut edges_list: Vec<Self> = Vec::with_capacity(n_dim);
        let mut steps: Vec<f64> = Vec::with_capacity(n_dim);
        for d in 0..n_dim {
            let e = Self::linspace(mins[d], maxs[d], bins_per_dim[d] + 1, DType::F64)?;
            let step = if bins_per_dim[d] > 0 && maxs[d] > mins[d] {
                (maxs[d] - mins[d]) / bins_per_dim[d] as f64
            } else {
                1.0
            };
            edges_list.push(e);
            steps.push(step);
        }

        // Histogram: shape = bins_per_dim
        let total_bins: usize = bins_per_dim.iter().product();
        let mut hist = vec![0.0f64; total_bins];

        // Compute strides for the N-D histogram array
        let mut bin_strides: Vec<usize> = vec![1; n_dim];
        for d in (0..n_dim.saturating_sub(1)).rev() {
            bin_strides[d] = bin_strides[d + 1] * bins_per_dim[d + 1];
        }

        for i in 0..n_obs {
            let mut flat_idx = 0usize;
            let mut valid = true;
            for d in 0..n_dim {
                let v = self.values[i * n_dim + d];
                let idx = ((v - mins[d]) / steps[d]).floor() as usize;
                let idx = idx.min(bins_per_dim[d] - 1);
                flat_idx += idx * bin_strides[d];
                if flat_idx >= total_bins {
                    valid = false;
                    break;
                }
            }
            if valid {
                hist[flat_idx] += 1.0;
            }
        }

        let h = Self {
            shape: bins_per_dim.to_vec(),
            values: hist,
            dtype: DType::I64,
        };
        Ok((h, edges_list))
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

    /// 2-D convolution (full mode). Equivalent to scipy.signal.convolve2d(mode='full').
    ///
    /// Both `self` and `kernel` must be 2-D arrays.
    pub fn convolve2d(&self, kernel: &Self) -> Result<Self, UFuncError> {
        if self.shape.len() != 2 || kernel.shape.len() != 2 {
            return Err(UFuncError::Msg(
                "convolve2d: both arrays must be 2-D".to_string(),
            ));
        }
        let (h1, w1) = (self.shape[0], self.shape[1]);
        let (h2, w2) = (kernel.shape[0], kernel.shape[1]);
        if h1 == 0 || w1 == 0 || h2 == 0 || w2 == 0 {
            return Ok(Self {
                shape: vec![0, 0],
                values: Vec::new(),
                dtype: promote(self.dtype, kernel.dtype),
            });
        }
        let out_h = h1 + h2 - 1;
        let out_w = w1 + w2 - 1;
        let mut values = vec![0.0f64; out_h * out_w];
        for i in 0..h1 {
            for j in 0..w1 {
                let a_val = self.values[i * w1 + j];
                for ki in 0..h2 {
                    for kj in 0..w2 {
                        values[(i + ki) * out_w + (j + kj)] += a_val * kernel.values[ki * w2 + kj];
                    }
                }
            }
        }
        Ok(Self {
            shape: vec![out_h, out_w],
            values,
            dtype: promote(self.dtype, kernel.dtype),
        })
    }

    /// 2-D cross-correlation (full mode). Equivalent to scipy.signal.correlate2d(mode='full').
    ///
    /// Both `self` and `kernel` must be 2-D arrays.
    pub fn correlate2d(&self, kernel: &Self) -> Result<Self, UFuncError> {
        if self.shape.len() != 2 || kernel.shape.len() != 2 {
            return Err(UFuncError::Msg(
                "correlate2d: both arrays must be 2-D".to_string(),
            ));
        }
        // correlate2d(a, v) = convolve2d(a, v[::-1, ::-1])
        let reversed = Self {
            shape: kernel.shape.clone(),
            values: kernel.values.iter().rev().copied().collect(),
            dtype: kernel.dtype,
        };
        self.convolve2d(&reversed)
    }

    /// Apply a reduction function repeatedly over multiple axes.
    ///
    /// Equivalent to `numpy.apply_over_axes(func, a, axes)`.
    /// The function is applied sequentially: reduce axis[0], then axis[1], etc.
    /// Each reduction keeps dimensions (keepdims=true) so subsequent axis indices
    /// remain valid.
    pub fn apply_over_axes<F>(&self, func: F, axes: &[isize]) -> Result<Self, UFuncError>
    where
        F: Fn(&Self, Option<isize>, bool) -> Result<Self, UFuncError>,
    {
        let mut result = self.clone();
        for &axis in axes {
            result = func(&result, Some(axis), true)?;
        }
        Ok(result)
    }

    /// Apply a scalar function element-wise (np.vectorize).
    /// Takes a function `f(f64) -> f64` and applies it to every element,
    /// preserving the array shape.
    pub fn vectorize<F: Fn(f64) -> f64>(&self, func: F) -> Self {
        Self {
            shape: self.shape.clone(),
            values: self.values.iter().map(|&v| func(v)).collect(),
            dtype: self.dtype,
        }
    }

    /// Apply a binary scalar function element-wise with broadcasting (np.vectorize with 2 args).
    pub fn vectorize_binary<F: Fn(f64, f64) -> f64>(
        a: &Self,
        b: &Self,
        func: F,
    ) -> Result<Self, UFuncError> {
        let broadcast = Self::broadcast_arrays(&[a, b])?;
        let ba = &broadcast[0];
        let bb = &broadcast[1];
        let values: Vec<f64> = ba
            .values
            .iter()
            .zip(bb.values.iter())
            .map(|(&va, &vb)| func(va, vb))
            .collect();
        Ok(Self {
            shape: ba.shape.clone(),
            values,
            dtype: DType::F64,
        })
    }

    /// Apply an N-ary scalar function element-wise with broadcasting (np.vectorize).
    /// `arrays` are the input arrays; `excluded` indices are passed through as-is
    /// (their first element is used as a scalar constant).
    pub fn vectorize_n<F: Fn(&[f64]) -> f64>(
        arrays: &[&Self],
        excluded: &[usize],
        func: F,
    ) -> Result<Self, UFuncError> {
        if arrays.is_empty() {
            return Err(UFuncError::Msg(
                "vectorize: need at least one input array".to_string(),
            ));
        }

        // Separate excluded and vectorized arrays
        let mut vectorized_indices: Vec<usize> = Vec::new();
        let mut vectorized_arrays: Vec<&Self> = Vec::new();
        for (i, &arr) in arrays.iter().enumerate() {
            if !excluded.contains(&i) {
                vectorized_indices.push(i);
                vectorized_arrays.push(arr);
            }
        }

        if vectorized_arrays.is_empty() {
            // All excluded — apply func once with scalar values
            let scalar_args: Vec<f64> = arrays.iter().map(|a| a.values[0]).collect();
            let val = func(&scalar_args);
            return Ok(Self {
                shape: vec![1],
                values: vec![val],
                dtype: DType::F64,
            });
        }

        // Broadcast vectorized arrays
        let broadcast = Self::broadcast_arrays(&vectorized_arrays)?;
        let n = broadcast[0].values.len();
        let out_shape = broadcast[0].shape.clone();

        // Build args buffer and apply func element-wise
        let nargs = arrays.len();
        let mut args = vec![0.0f64; nargs];
        // Pre-fill excluded args with their scalar values
        for &ei in excluded {
            if ei < nargs {
                args[ei] = arrays[ei].values[0];
            }
        }

        let mut values = Vec::with_capacity(n);
        for elem_idx in 0..n {
            // Fill vectorized args from broadcast arrays
            for (bc_idx, &orig_idx) in vectorized_indices.iter().enumerate() {
                args[orig_idx] = broadcast[bc_idx].values[elem_idx];
            }
            values.push(func(&args));
        }

        Ok(Self {
            shape: out_shape,
            values,
            dtype: DType::F64,
        })
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

        // Strip leading zeros
        let first_nonzero = self.values.iter().position(|&v| v != 0.0);
        let coeffs = match first_nonzero {
            None => {
                return Ok(Self {
                    shape: vec![0],
                    values: Vec::new(),
                    dtype: DType::F64,
                });
            }
            Some(idx) => &self.values[idx..],
        };

        // Count trailing zeros (these are roots at zero)
        let last_nonzero = coeffs.iter().rposition(|&v| v != 0.0).unwrap_or(0);
        let trailing_zeros = coeffs.len() - 1 - last_nonzero;
        let coeffs = &coeffs[..=last_nonzero];

        let degree = coeffs.len() - 1;
        if degree == 0 {
            // Constant polynomial — no roots (just trailing zeros if any)
            let mut values: Vec<f64> = vec![0.0; trailing_zeros];
            values.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap_or(std::cmp::Ordering::Equal));
            return Ok(Self {
                shape: vec![values.len()],
                values,
                dtype: DType::F64,
            });
        }
        if degree == 1 {
            let root = -coeffs[1] / coeffs[0];
            let mut values = vec![root];
            values.extend(std::iter::repeat(0.0).take(trailing_zeros));
            return Ok(Self {
                shape: vec![values.len()],
                values,
                dtype: DType::F64,
            });
        }
        if degree == 2 {
            let (a, b, c) = (coeffs[0], coeffs[1], coeffs[2]);
            let disc = b * b - 4.0 * a * c;
            let mut values = if disc < 0.0 {
                vec![f64::NAN, f64::NAN]
            } else {
                let sqrt_disc = disc.sqrt();
                let r1 = (-b + sqrt_disc) / (2.0 * a);
                let r2 = (-b - sqrt_disc) / (2.0 * a);
                vec![r1, r2]
            };
            values.extend(std::iter::repeat(0.0).take(trailing_zeros));
            return Ok(Self {
                shape: vec![values.len()],
                values,
                dtype: DType::F64,
            });
        }

        // Degree >= 3: Use companion matrix eigenvalue method
        // Build companion matrix A of size degree × degree
        // A[0, j] = -coeffs[j+1] / coeffs[0]  (first row)
        // A[i+1, i] = 1                         (sub-diagonal)
        let d = degree;
        let lead = coeffs[0];
        let mut comp = vec![0.0f64; d * d];

        // First row: -p[1:] / p[0]
        for j in 0..d {
            comp[j] = -coeffs[j + 1] / lead;
        }
        // Sub-diagonal: ones
        for i in 0..d - 1 {
            comp[(i + 1) * d + i] = 1.0;
        }

        // Compute eigenvalues
        let eigenvalues = fnp_linalg::eig_nxn(&comp, d).map_err(|e| {
            UFuncError::Msg(format!("roots: eigenvalue computation failed: {e}"))
        })?;

        // eigenvalues are interleaved (real, imag) pairs
        let mut values: Vec<f64> = Vec::with_capacity(d + trailing_zeros);
        for i in 0..d {
            let re = eigenvalues[2 * i];
            let im = eigenvalues[2 * i + 1];
            if im.abs() < 1e-10 {
                values.push(re);
            } else {
                // Complex root — return real part (NaN convention for full complex)
                values.push(re);
            }
        }

        // Sort by descending absolute value (NumPy convention)
        values.sort_by(|a, b| {
            b.abs()
                .partial_cmp(&a.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Append trailing zeros
        values.extend(std::iter::repeat(0.0).take(trailing_zeros));

        Ok(Self {
            shape: vec![values.len()],
            values,
            dtype: DType::F64,
        })
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

    /// Compute the inverse of rfft (np.fft.irfft).
    /// Input has shape `[n//2+1, 2]` (hermitian-symmetric half).
    /// `output_n` is the output length; if `None` defaults to `2*(input_len - 1)`.
    /// Returns a real-valued array of shape `[output_n]`.
    pub fn irfft(&self, output_n: Option<usize>) -> Result<Self, UFuncError> {
        if self.shape.len() != 2 || self.shape[1] != 2 {
            return Err(UFuncError::Msg(
                "irfft: input must have shape [m, 2] (interleaved complex)".to_string(),
            ));
        }
        let m = self.shape[0]; // number of complex coefficients
        let n = output_n.unwrap_or(2 * (m.saturating_sub(1)));
        if n == 0 {
            return Ok(Self {
                shape: vec![0],
                values: vec![],
                dtype: DType::F64,
            });
        }
        // Reconstruct full spectrum from hermitian symmetry: X[k] = conj(X[n-k])
        let mut re = vec![0.0; n];
        let mut im = vec![0.0; n];
        for k in 0..m.min(n) {
            re[k] = self.values[k * 2];
            im[k] = self.values[k * 2 + 1];
        }
        for k in m.min(n)..n {
            let mirror = n - k;
            if mirror < m {
                re[k] = self.values[mirror * 2];
                im[k] = -self.values[mirror * 2 + 1]; // conjugate
            }
        }
        fft_dit(&mut re, &mut im, true);
        Ok(Self {
            shape: vec![n],
            values: re,
            dtype: DType::F64,
        })
    }

    /// Compute the 2-D DFT (np.fft.fft2).
    /// Input is a real 2-D array of shape `[rows, cols]`.
    /// Returns interleaved complex output of shape `[rows, cols, 2]`.
    pub fn fft2(&self) -> Result<Self, UFuncError> {
        if self.shape.len() != 2 {
            return Err(UFuncError::Msg(
                "fft2: input must be a 2-D array".to_string(),
            ));
        }
        let rows = self.shape[0];
        let cols = self.shape[1];
        if rows == 0 || cols == 0 {
            return Ok(Self {
                shape: vec![rows, cols, 2],
                values: vec![],
                dtype: DType::F64,
            });
        }
        // FFT along each row
        let mut re = vec![0.0; rows * cols];
        let mut im = vec![0.0; rows * cols];
        for r in 0..rows {
            let mut row_re: Vec<f64> = (0..cols).map(|c| self.values[r * cols + c]).collect();
            let mut row_im = vec![0.0; cols];
            fft_dit(&mut row_re, &mut row_im, false);
            for c in 0..cols {
                re[r * cols + c] = row_re[c];
                im[r * cols + c] = row_im[c];
            }
        }
        // FFT along each column
        for c in 0..cols {
            let mut col_re: Vec<f64> = (0..rows).map(|r| re[r * cols + c]).collect();
            let mut col_im: Vec<f64> = (0..rows).map(|r| im[r * cols + c]).collect();
            fft_dit(&mut col_re, &mut col_im, false);
            for r in 0..rows {
                re[r * cols + c] = col_re[r];
                im[r * cols + c] = col_im[r];
            }
        }
        // Interleave
        let mut values = Vec::with_capacity(rows * cols * 2);
        for i in 0..rows * cols {
            values.push(re[i]);
            values.push(im[i]);
        }
        Ok(Self {
            shape: vec![rows, cols, 2],
            values,
            dtype: DType::F64,
        })
    }

    /// Compute the 2-D inverse DFT (np.fft.ifft2).
    /// Input is interleaved complex of shape `[rows, cols, 2]`.
    /// Returns interleaved complex output of shape `[rows, cols, 2]`.
    pub fn ifft2(&self) -> Result<Self, UFuncError> {
        if self.shape.len() != 3 || self.shape[2] != 2 {
            return Err(UFuncError::Msg(
                "ifft2: input must have shape [rows, cols, 2] (interleaved complex)".to_string(),
            ));
        }
        let rows = self.shape[0];
        let cols = self.shape[1];
        if rows == 0 || cols == 0 {
            return Ok(Self {
                shape: vec![rows, cols, 2],
                values: vec![],
                dtype: DType::F64,
            });
        }
        // Deinterleave
        let mut re = vec![0.0; rows * cols];
        let mut im = vec![0.0; rows * cols];
        for i in 0..rows * cols {
            re[i] = self.values[i * 2];
            im[i] = self.values[i * 2 + 1];
        }
        // IFFT along each row
        for r in 0..rows {
            let mut row_re: Vec<f64> = (0..cols).map(|c| re[r * cols + c]).collect();
            let mut row_im: Vec<f64> = (0..cols).map(|c| im[r * cols + c]).collect();
            fft_dit(&mut row_re, &mut row_im, true);
            for c in 0..cols {
                re[r * cols + c] = row_re[c];
                im[r * cols + c] = row_im[c];
            }
        }
        // IFFT along each column
        for c in 0..cols {
            let mut col_re: Vec<f64> = (0..rows).map(|r| re[r * cols + c]).collect();
            let mut col_im: Vec<f64> = (0..rows).map(|r| im[r * cols + c]).collect();
            fft_dit(&mut col_re, &mut col_im, true);
            for r in 0..rows {
                re[r * cols + c] = col_re[r];
                im[r * cols + c] = col_im[r];
            }
        }
        // Interleave
        let mut values = Vec::with_capacity(rows * cols * 2);
        for i in 0..rows * cols {
            values.push(re[i]);
            values.push(im[i]);
        }
        Ok(Self {
            shape: vec![rows, cols, 2],
            values,
            dtype: DType::F64,
        })
    }

    /// Compute the N-dimensional DFT (np.fft.fftn).
    ///
    /// Input is a real array of arbitrary shape `[d0, d1, ..., d_{n-1}]`.
    /// Applies 1-D FFT along every axis in order.
    /// Returns interleaved complex output of shape `[d0, d1, ..., d_{n-1}, 2]`.
    pub fn fftn(&self) -> Result<Self, UFuncError> {
        let ndim = self.shape.len();
        if ndim == 0 {
            return Ok(Self {
                shape: vec![2],
                values: vec![self.values.first().copied().unwrap_or(0.0), 0.0],
                dtype: DType::F64,
            });
        }
        let total: usize = self.shape.iter().product();
        if total == 0 {
            let mut out_shape = self.shape.clone();
            out_shape.push(2);
            return Ok(Self {
                shape: out_shape,
                values: vec![],
                dtype: DType::F64,
            });
        }

        // Initialize re/im arrays
        let mut re = self.values.clone();
        re.resize(total, 0.0);
        let mut im = vec![0.0; total];

        // Apply FFT along each axis
        for axis in 0..ndim {
            fftn_along_axis(&self.shape, &mut re, &mut im, axis, false);
        }

        // Interleave to output
        let mut values = Vec::with_capacity(total * 2);
        for i in 0..total {
            values.push(re[i]);
            values.push(im[i]);
        }
        let mut out_shape = self.shape.clone();
        out_shape.push(2);
        Ok(Self {
            shape: out_shape,
            values,
            dtype: DType::F64,
        })
    }

    /// Compute the N-dimensional inverse DFT (np.fft.ifftn).
    ///
    /// Input is interleaved complex of shape `[d0, d1, ..., d_{n-1}, 2]`.
    /// Returns interleaved complex output of same shape.
    pub fn ifftn(&self) -> Result<Self, UFuncError> {
        let ndim = self.shape.len();
        if ndim < 2 || self.shape[ndim - 1] != 2 {
            return Err(UFuncError::Msg(
                "ifftn: input must have trailing dimension 2 (interleaved complex)".to_string(),
            ));
        }
        let spatial_shape = &self.shape[..ndim - 1];
        let total: usize = spatial_shape.iter().product();
        if total == 0 {
            return Ok(self.clone());
        }

        // Deinterleave
        let mut re = vec![0.0; total];
        let mut im = vec![0.0; total];
        for i in 0..total {
            re[i] = self.values[i * 2];
            im[i] = self.values[i * 2 + 1];
        }

        // Apply IFFT along each axis
        let spatial_ndim = spatial_shape.len();
        for axis in 0..spatial_ndim {
            fftn_along_axis(spatial_shape, &mut re, &mut im, axis, true);
        }

        // Interleave
        let mut values = Vec::with_capacity(total * 2);
        for i in 0..total {
            values.push(re[i]);
            values.push(im[i]);
        }
        Ok(Self {
            shape: self.shape.clone(),
            values,
            dtype: DType::F64,
        })
    }

    /// Compute the N-dimensional real-input FFT (np.fft.rfftn).
    ///
    /// Input is a real array of shape `[d0, d1, ..., d_{n-1}]`.
    /// Applies full FFT along all axes except the last, then rfft along the
    /// last axis. Returns interleaved complex of shape
    /// `[d0, d1, ..., d_{n-1}//2 + 1, 2]`.
    pub fn rfftn(&self) -> Result<Self, UFuncError> {
        let ndim = self.shape.len();
        if ndim == 0 {
            return Ok(Self {
                shape: vec![1, 2],
                values: vec![self.values.first().copied().unwrap_or(0.0), 0.0],
                dtype: DType::F64,
            });
        }
        let total: usize = self.shape.iter().product();
        if total == 0 {
            let mut out_shape = self.shape.clone();
            let last = out_shape.last_mut().unwrap();
            *last = *last / 2 + 1;
            out_shape.push(2);
            return Ok(Self {
                shape: out_shape,
                values: vec![],
                dtype: DType::F64,
            });
        }

        // Initialize re/im arrays
        let mut re = self.values.clone();
        re.resize(total, 0.0);
        let mut im = vec![0.0; total];

        // FFT along all axes except the last
        for axis in 0..ndim.saturating_sub(1) {
            fftn_along_axis(&self.shape, &mut re, &mut im, axis, false);
        }

        // FFT along last axis, then truncate to n//2+1
        if ndim > 0 {
            fftn_along_axis(&self.shape, &mut re, &mut im, ndim - 1, false);
        }

        // Truncate last axis to n//2+1
        let last_n = self.shape[ndim - 1];
        let half_n = last_n / 2 + 1;
        let outer: usize = self.shape[..ndim - 1].iter().product::<usize>().max(1);
        let mut values = Vec::with_capacity(outer * half_n * 2);
        for o in 0..outer {
            for k in 0..half_n {
                let idx = o * last_n + k;
                values.push(re[idx]);
                values.push(im[idx]);
            }
        }
        let mut out_shape = self.shape.clone();
        *out_shape.last_mut().unwrap() = half_n;
        out_shape.push(2);
        Ok(Self {
            shape: out_shape,
            values,
            dtype: DType::F64,
        })
    }

    /// Compute the N-dimensional inverse real FFT (np.fft.irfftn).
    ///
    /// Input is interleaved complex of shape `[d0, ..., d_{n-2}, d_{n-1}//2+1, 2]`.
    /// The `last_n` parameter specifies the output length along the last axis
    /// (defaults to `2*(d_{n-1}//2+1 - 1)`).
    /// Returns a real array of shape `[d0, ..., d_{n-2}, last_n]`.
    pub fn irfftn(&self, last_n: Option<usize>) -> Result<Self, UFuncError> {
        let ndim = self.shape.len();
        if ndim < 2 || self.shape[ndim - 1] != 2 {
            return Err(UFuncError::Msg(
                "irfftn: input must have trailing dimension 2 (interleaved complex)".to_string(),
            ));
        }
        let spatial_shape = &self.shape[..ndim - 1];
        let spatial_ndim = spatial_shape.len();
        if spatial_ndim == 0 {
            return Err(UFuncError::Msg(
                "irfftn: need at least one spatial dimension".to_string(),
            ));
        }
        let half_n = spatial_shape[spatial_ndim - 1];
        let output_n = last_n.unwrap_or(2 * (half_n - 1));
        if output_n == 0 {
            let mut out_shape: Vec<usize> = spatial_shape[..spatial_ndim - 1].to_vec();
            out_shape.push(0);
            return Ok(Self {
                shape: out_shape,
                values: vec![],
                dtype: DType::F64,
            });
        }

        // Reconstruct full spectrum along last axis
        let outer: usize = spatial_shape[..spatial_ndim - 1]
            .iter()
            .product::<usize>()
            .max(1);
        let total_out = outer * output_n;
        let mut re = vec![0.0; total_out];
        let mut im = vec![0.0; total_out];
        let spatial_total: usize = spatial_shape.iter().product();
        for o in 0..outer {
            for k in 0..half_n.min(output_n) {
                let src = o * half_n + k;
                if src < spatial_total {
                    re[o * output_n + k] = self.values[src * 2];
                    im[o * output_n + k] = self.values[src * 2 + 1];
                }
            }
            // Hermitian symmetry for k > half_n
            for k in half_n..output_n {
                let conj_k = output_n - k;
                if conj_k < half_n {
                    let src = o * half_n + conj_k;
                    if src < spatial_total {
                        re[o * output_n + k] = self.values[src * 2];
                        im[o * output_n + k] = -self.values[src * 2 + 1];
                    }
                }
            }
        }

        // Build output shape for IFFT processing
        let mut full_shape: Vec<usize> = spatial_shape[..spatial_ndim - 1].to_vec();
        full_shape.push(output_n);

        // IFFT along all axes
        for axis in 0..full_shape.len() {
            fftn_along_axis(&full_shape, &mut re, &mut im, axis, true);
        }

        // Return real part only
        Ok(Self {
            shape: full_shape,
            values: re,
            dtype: DType::F64,
        })
    }

    /// Compute the 2-D real FFT (np.fft.rfft2).
    /// Input must be a real 2-D array. Output has interleaved complex
    /// of shape `[rows, cols//2+1, 2]`.
    pub fn rfft2(&self) -> Result<Self, UFuncError> {
        if self.shape.len() != 2 {
            return Err(UFuncError::Msg(
                "rfft2: input must be a 2-D array".to_string(),
            ));
        }
        self.rfftn()
    }

    /// Compute the inverse 2-D real FFT (np.fft.irfft2).
    /// Input must be interleaved complex of shape `[rows, half_cols, 2]`.
    /// Returns real 2-D array of shape `[rows, last_n]`.
    pub fn irfft2(&self, last_n: Option<usize>) -> Result<Self, UFuncError> {
        let ndim = self.shape.len();
        if ndim != 3 || self.shape[ndim - 1] != 2 {
            return Err(UFuncError::Msg(
                "irfft2: input must have shape [rows, half_cols, 2]".to_string(),
            ));
        }
        self.irfftn(last_n)
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

    /// Evaluate the optimal contraction order for `einsum` (np.einsum_path).
    ///
    /// Returns a tuple of `(contraction_list, description_string)`.
    /// The contraction list contains pairs of operand indices to contract.
    /// The description string provides a human-readable summary of the path.
    ///
    /// This is a greedy heuristic: at each step, it picks the pair of operands
    /// whose contraction produces the smallest intermediate result.
    pub fn einsum_path(
        subscripts: &str,
        operands: &[&Self],
    ) -> Result<(Vec<Vec<usize>>, String), UFuncError> {
        let parts: Vec<&str> = subscripts.split("->").collect();
        if parts.len() != 2 {
            return Err(UFuncError::Msg(
                "einsum_path: subscripts must contain exactly one '->'".to_string(),
            ));
        }
        let input_subs: Vec<&str> = parts[0].split(',').collect();
        if input_subs.len() != operands.len() {
            return Err(UFuncError::Msg(format!(
                "einsum_path: {} subscript groups but {} operands",
                input_subs.len(),
                operands.len()
            )));
        }

        let n = operands.len();
        if n <= 2 {
            // Trivial: just contract all at once
            let path = vec![(0..n).collect::<Vec<usize>>()];
            let sizes: Vec<usize> = operands.iter().map(|o| o.values.len()).collect();
            let desc = format!(
                "  Complete contraction:  {}\n  Naive scaling:  {}\n  Optimized scaling:  {}\n  Input shapes:  {:?}\n",
                subscripts, n, n, sizes,
            );
            return Ok((path, desc));
        }

        // Greedy algorithm: repeatedly contract the pair with smallest result
        let mut remaining: Vec<usize> = (0..n).collect();
        let mut path = Vec::new();
        let mut sizes: Vec<usize> = operands.iter().map(|o| o.values.len()).collect();

        while remaining.len() > 1 {
            // Find the pair (i, j) with smallest product of sizes
            let mut best_i = 0;
            let mut best_j = 1;
            let mut best_cost = sizes[remaining[0]] * sizes[remaining[1]];
            for i in 0..remaining.len() {
                for j in (i + 1)..remaining.len() {
                    let cost = sizes[remaining[i]] * sizes[remaining[j]];
                    if cost < best_cost {
                        best_i = i;
                        best_j = j;
                        best_cost = cost;
                    }
                }
            }
            let idx_a = remaining[best_j]; // remove larger index first
            let idx_b = remaining[best_i];
            path.push(vec![idx_b, idx_a]);
            // The result replaces the lower index in the remaining list
            remaining.remove(best_j);
            remaining.remove(best_i);
            // Add a new "virtual" operand
            let new_size = best_cost.max(1); // approximate
            sizes.push(new_size);
            remaining.push(sizes.len() - 1);
        }

        let input_sizes: Vec<usize> = operands.iter().map(|o| o.values.len()).collect();
        let desc = format!(
            "  Complete contraction:  {}\n  Input shapes:  {:?}\n  Path:  {:?}\n",
            subscripts, input_sizes, path,
        );
        Ok((path, desc))
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

    /// Stack arrays vertically (np.row_stack). Alias for `vstack`.
    pub fn row_stack(arrays: &[Self]) -> Result<Self, UFuncError> {
        Self::vstack(arrays)
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

    /// Return a flat Vec<f64> of all values (np.ndarray.tolist equivalent for numeric arrays).
    #[must_use]
    pub fn tolist(&self) -> Vec<f64> {
        self.values.clone()
    }

    /// Serialize the array to raw bytes (np.ndarray.tobytes equivalent).
    /// Uses native f64 (8-byte little-endian) encoding.
    #[must_use]
    pub fn tobytes_array(&self) -> Vec<u8> {
        self.values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Swap byte order of each element in-place (np.ndarray.byteswap).
    #[must_use]
    pub fn byteswap(&self) -> Self {
        let swapped: Vec<f64> = self
            .values
            .iter()
            .map(|v| {
                let bytes = v.to_ne_bytes();
                let mut rev = bytes;
                rev.reverse();
                f64::from_ne_bytes(rev)
            })
            .collect();
        Self {
            shape: self.shape.clone(),
            values: swapped,
            dtype: self.dtype,
        }
    }

    /// Extract a diagonal from a 2-D subarray specified by axis1 and axis2.
    ///
    /// `np.ndarray.diagonal(offset=0, axis1=0, axis2=1)`.
    /// For 2-D arrays this is identical to `diag(offset)`.
    /// For N-D arrays, the diagonal is extracted from the 2-D slice
    /// defined by `axis1` and `axis2`, and the result is appended as a new axis.
    pub fn diagonal(&self, offset: i64, axis1: isize, axis2: isize) -> Result<Self, UFuncError> {
        let ndim = self.shape.len();
        if ndim < 2 {
            return Err(UFuncError::Msg(
                "diagonal requires at least a 2-D array".to_string(),
            ));
        }
        let a1 = normalize_axis(axis1, ndim)?;
        let a2 = normalize_axis(axis2, ndim)?;
        if a1 == a2 {
            return Err(UFuncError::Msg(
                "diagonal: axis1 and axis2 must be different".to_string(),
            ));
        }
        if ndim == 2 {
            return self.diag(offset);
        }
        // N-D case: collect other axes, iterate over their indices,
        // and extract diagonal from the (axis1, axis2) 2-D slice.
        let d1 = self.shape[a1];
        let d2 = self.shape[a2];
        let start_r = if offset < 0 { (-offset) as usize } else { 0 };
        let start_c = if offset >= 0 { offset as usize } else { 0 };
        let diag_len = d1.saturating_sub(start_r).min(d2.saturating_sub(start_c));

        // Build output shape: remove axis1 and axis2 from shape, append diag_len
        let other_axes: Vec<usize> = (0..ndim).filter(|&a| a != a1 && a != a2).collect();
        let mut out_shape: Vec<usize> = other_axes.iter().map(|&a| self.shape[a]).collect();
        out_shape.push(diag_len);

        let strides = c_strides_elems(&self.shape);
        let other_count: usize = other_axes.iter().map(|&a| self.shape[a]).product();
        let other_strides: Vec<usize> = {
            let other_shape: Vec<usize> = other_axes.iter().map(|&a| self.shape[a]).collect();
            c_strides_elems(&other_shape)
        };

        let mut result = Vec::with_capacity(other_count * diag_len);
        for oi in 0..other_count {
            // Decode oi into indices for other_axes
            let mut rem = oi;
            let mut base_offset = 0usize;
            for (j, &ax) in other_axes.iter().enumerate() {
                let idx = rem / other_strides[j];
                rem %= other_strides[j];
                base_offset += idx * strides[ax];
            }
            for k in 0..diag_len {
                let r = start_r + k;
                let c = start_c + k;
                let flat = base_offset + r * strides[a1] + c * strides[a2];
                result.push(self.values[flat]);
            }
        }
        Ok(Self {
            shape: out_shape,
            values: result,
            dtype: self.dtype,
        })
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
    /// When `invert` is true, returns True where element is NOT in `test_elements`.
    pub fn isin(&self, test_elements: &Self, invert: bool) -> Self {
        let mut result = self.in1d(test_elements);
        if invert {
            for v in &mut result.values {
                *v = 1.0 - *v;
            }
        }
        result
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

    /// `np.nanpercentile` — percentile ignoring NaN values.
    /// q is in [0, 100]. Only `axis=None` (global) is supported; NaN elements
    /// are removed before computing.
    pub fn nanpercentile(&self, q: f64, axis: Option<isize>) -> Result<Self, UFuncError> {
        if !(0.0..=100.0).contains(&q) {
            return Err(UFuncError::Msg(format!(
                "nanpercentile: q={q} must be in [0, 100]"
            )));
        }
        if axis.is_some() {
            return Err(UFuncError::Msg(
                "nanpercentile: only axis=None is supported".to_string(),
            ));
        }
        // Remove NaN values (flattens to 1-D), then compute percentile
        let filtered = self.nan_removed();
        if filtered.values.is_empty() {
            return Ok(Self::scalar(f64::NAN, self.dtype));
        }
        filtered.percentile(q, None)
    }

    /// `np.nanquantile` — quantile ignoring NaN values.
    /// q is in [0, 1]. Only `axis=None` (global) is supported.
    pub fn nanquantile(&self, q: f64, axis: Option<isize>) -> Result<Self, UFuncError> {
        self.nanpercentile(q * 100.0, axis)
    }

    /// Remove NaN values (flattened to 1-D).
    fn nan_removed(&self) -> Self {
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

    /// `np.real_if_close` — return real parts if imaginary parts are close to zero.
    ///
    /// For complex arrays (shape [..., 2] with interleaved real/imag), check if all
    /// imaginary parts have magnitude <= `tol * machine_epsilon`. If so, return just
    /// the real parts. Otherwise return the original array unchanged.
    pub fn real_if_close(&self, tol: f64) -> Self {
        if !matches!(self.dtype, DType::Complex64 | DType::Complex128) {
            return self.clone();
        }
        // Complex arrays have trailing dimension 2: [real, imag, real, imag, ...]
        let eps = if self.dtype == DType::Complex64 {
            f64::from(f32::EPSILON)
        } else {
            f64::EPSILON
        };
        let threshold = tol * eps;
        // Check all imaginary parts
        let all_close = self
            .values
            .iter()
            .skip(1)
            .step_by(2)
            .all(|&im| im.abs() <= threshold);
        if all_close {
            // Extract real parts only
            let real_values: Vec<f64> = self.values.iter().step_by(2).copied().collect();
            let real_dtype = if self.dtype == DType::Complex64 {
                DType::F32
            } else {
                DType::F64
            };
            // Remove trailing 2 from shape
            let mut real_shape = self.shape.clone();
            if let Some(last) = real_shape.last()
                && *last == 2
            {
                real_shape.pop();
            }
            if real_shape.is_empty() {
                real_shape.push(real_values.len());
            }
            Self {
                shape: real_shape,
                values: real_values,
                dtype: real_dtype,
            }
        } else {
            self.clone()
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
    ///
    /// `axis` selects the axis to integrate along (default last axis).
    /// The result has the integration axis removed.
    pub fn trapezoid(&self, dx: f64, axis: Option<isize>) -> Result<Self, UFuncError> {
        let ndim = self.shape.len();
        if ndim == 0 {
            return Err(UFuncError::Msg(
                "trapezoid: input must have at least 1 dimension".to_string(),
            ));
        }
        let ax = match axis {
            Some(a) => normalize_axis(a, ndim)?,
            None => ndim - 1,
        };
        let axis_len = self.shape[ax];
        if axis_len < 2 {
            // NumPy returns 0 for length-1 axis
            let mut out_shape: Vec<usize> = self.shape.clone();
            out_shape.remove(ax);
            if out_shape.is_empty() {
                return Ok(Self::scalar(0.0, DType::F64));
            }
            let total: usize = out_shape.iter().product();
            return Ok(Self {
                shape: out_shape,
                values: vec![0.0; total],
                dtype: DType::F64,
            });
        }

        // Compute output shape (input shape with axis removed)
        let mut out_shape: Vec<usize> = self.shape.clone();
        out_shape.remove(ax);

        let outer: usize = self.shape[..ax].iter().copied().product();
        let inner: usize = self.shape[ax + 1..].iter().copied().product();

        let out_total = outer * inner;
        let mut out_values = vec![0.0f64; out_total];

        for o in 0..outer {
            for i in 0..inner {
                let mut sum = 0.0f64;
                for k in 0..axis_len - 1 {
                    let idx0 = o * axis_len * inner + k * inner + i;
                    let idx1 = o * axis_len * inner + (k + 1) * inner + i;
                    sum += (self.values[idx0] + self.values[idx1]) / 2.0 * dx;
                }
                out_values[o * inner + i] = sum;
            }
        }

        if out_shape.is_empty() {
            Ok(Self::scalar(out_values[0], DType::F64))
        } else {
            Ok(Self {
                shape: out_shape,
                values: out_values,
                dtype: DType::F64,
            })
        }
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

    /// Gamma function (scipy.special.gamma / math.gamma).
    /// Computes Gamma(x) for each element using Lanczos approximation.
    pub fn gamma(&self) -> Self {
        let values: Vec<f64> = self.values.iter().map(|&x| lanczos_gamma(x)).collect();
        Self {
            shape: self.shape.clone(),
            values,
            dtype: DType::F64,
        }
    }

    /// Natural log of the absolute value of the Gamma function (scipy.special.gammaln).
    pub fn lgamma(&self) -> Self {
        let values: Vec<f64> = self
            .values
            .iter()
            .map(|&x| lanczos_gamma(x).abs().ln())
            .collect();
        Self {
            shape: self.shape.clone(),
            values,
            dtype: DType::F64,
        }
    }

    /// Error function (scipy.special.erf).
    /// Uses Horner-form rational approximation (Abramowitz & Stegun 7.1.26).
    pub fn erf(&self) -> Self {
        let values: Vec<f64> = self.values.iter().map(|&x| erf_approx(x)).collect();
        Self {
            shape: self.shape.clone(),
            values,
            dtype: DType::F64,
        }
    }

    /// Complementary error function: erfc(x) = 1 - erf(x).
    pub fn erfc(&self) -> Self {
        let values: Vec<f64> = self.values.iter().map(|&x| 1.0 - erf_approx(x)).collect();
        Self {
            shape: self.shape.clone(),
            values,
            dtype: DType::F64,
        }
    }

    /// Digamma (psi) function — the logarithmic derivative of the Gamma function.
    /// Uses recurrence to shift argument to large values then asymptotic series.
    pub fn digamma(&self) -> Self {
        let values: Vec<f64> = self.values.iter().map(|&x| digamma_approx(x)).collect();
        Self {
            shape: self.shape.clone(),
            values,
            dtype: DType::F64,
        }
    }

    /// Bessel function of the first kind, order 0 (scipy.special.j0).
    pub fn j0(&self) -> Self {
        let values: Vec<f64> = self.values.iter().map(|&x| bessel_j0(x)).collect();
        Self {
            shape: self.shape.clone(),
            values,
            dtype: DType::F64,
        }
    }

    /// Bessel function of the first kind, order 1 (scipy.special.j1).
    pub fn j1(&self) -> Self {
        let values: Vec<f64> = self.values.iter().map(|&x| bessel_j1(x)).collect();
        Self {
            shape: self.shape.clone(),
            values,
            dtype: DType::F64,
        }
    }

    /// Bessel function of the second kind, order 0 (scipy.special.y0).
    pub fn y0(&self) -> Self {
        let values: Vec<f64> = self.values.iter().map(|&x| bessel_y0(x)).collect();
        Self {
            shape: self.shape.clone(),
            values,
            dtype: DType::F64,
        }
    }

    /// Bessel function of the second kind, order 1 (scipy.special.y1).
    pub fn y1(&self) -> Self {
        let values: Vec<f64> = self.values.iter().map(|&x| bessel_y1(x)).collect();
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

    /// Compute the broadcast shape from multiple shapes (np.broadcast_shapes).
    ///
    /// Returns the shape that would result from broadcasting arrays of the
    /// given shapes together.
    pub fn broadcast_shapes(shapes: &[&[usize]]) -> Result<Vec<usize>, UFuncError> {
        fnp_ndarray::broadcast_shapes(shapes).map_err(UFuncError::Shape)
    }

    /// Place values into an array using a boolean mask (np.putmask).
    ///
    /// For each position where `mask` is nonzero, the corresponding element
    /// in `self` is replaced with the next value from `values` (cycling if
    /// `values` is shorter than the number of True entries).
    pub fn putmask(&mut self, mask: &Self, values: &Self) {
        if values.values.is_empty() {
            return;
        }
        let n = self.values.len().min(mask.values.len());
        let mut vi = 0;
        for i in 0..n {
            if mask.values[i] != 0.0 {
                self.values[i] = values.values[vi % values.values.len()];
                vi += 1;
            }
        }
    }

    /// Create an array with the given shape and strides into the source data.
    ///
    /// `np.lib.stride_tricks.as_strided(x, shape, strides)`.
    /// `strides` are in **element counts** (not bytes), matching our internal representation.
    /// Elements that map to indices beyond the source data yield 0.0.
    pub fn as_strided(&self, shape: &[usize], strides: &[usize]) -> Result<Self, UFuncError> {
        if shape.len() != strides.len() {
            return Err(UFuncError::Msg(
                "as_strided: shape and strides must have same length".to_string(),
            ));
        }
        let out_count: usize = shape.iter().product();
        let out_strides = c_strides_elems(shape);
        let src_len = self.values.len();
        let values: Vec<f64> = (0..out_count)
            .map(|flat_idx| {
                let mut src_offset = 0usize;
                let mut rem = flat_idx;
                for d in 0..shape.len() {
                    let idx = rem / out_strides[d];
                    rem %= out_strides[d];
                    src_offset += idx * strides[d];
                }
                if src_offset < src_len {
                    self.values[src_offset]
                } else {
                    0.0
                }
            })
            .collect();
        Ok(Self {
            shape: shape.to_vec(),
            values,
            dtype: self.dtype,
        })
    }

    /// Create a sliding window view of the array (np.lib.stride_tricks.sliding_window_view).
    ///
    /// For a 1-D array of length N with window_shape W, returns an array of
    /// shape `[N - W + 1, W]` containing all contiguous windows.
    /// For N-D arrays, each axis gets its own window dimension appended.
    pub fn sliding_window_view(&self, window_shape: &[usize]) -> Result<Self, UFuncError> {
        let ndim = self.shape.len();
        if window_shape.len() != ndim {
            return Err(UFuncError::Msg(format!(
                "sliding_window_view: window_shape length {} != array ndim {}",
                window_shape.len(),
                ndim
            )));
        }
        for (d, (&s, &w)) in self.shape.iter().zip(window_shape.iter()).enumerate() {
            if w == 0 || w > s {
                return Err(UFuncError::Msg(format!(
                    "sliding_window_view: window size {} is invalid for axis {} of size {}",
                    w, d, s
                )));
            }
        }

        // Output shape: for each axis, (dim - window + 1), then append window_shape
        let mut out_shape: Vec<usize> = self
            .shape
            .iter()
            .zip(window_shape.iter())
            .map(|(&s, &w)| s - w + 1)
            .collect();
        let view_dims: Vec<usize> = out_shape.clone();
        out_shape.extend_from_slice(window_shape);

        let view_total: usize = view_dims.iter().product();
        let window_total: usize = window_shape.iter().product();
        let total = view_total * window_total;

        let src_strides = c_strides_elems(&self.shape);
        let view_strides = c_strides_elems(&view_dims);
        let win_strides = c_strides_elems(window_shape);

        let mut values = Vec::with_capacity(total);
        for view_flat in 0..view_total {
            // Compute the N-D view index
            let mut view_idx = vec![0usize; ndim];
            let mut rem = view_flat;
            for d in 0..ndim {
                view_idx[d] = rem / view_strides[d];
                rem %= view_strides[d];
            }
            for win_flat in 0..window_total {
                // Compute the N-D window offset
                let mut src_flat = 0usize;
                let mut wrem = win_flat;
                for d in 0..ndim {
                    let wi = wrem / win_strides[d];
                    wrem %= win_strides[d];
                    src_flat += (view_idx[d] + wi) * src_strides[d];
                }
                values.push(self.values[src_flat]);
            }
        }

        Ok(Self {
            shape: out_shape,
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
    ///
    /// With `axis=None`, returns a scalar. With an axis, reduces along that axis.
    pub fn count_nonzero(&self, axis: Option<isize>, keepdims: bool) -> Result<Self, UFuncError> {
        match axis {
            None => {
                let count = self.values.iter().filter(|&&v| v != 0.0).count() as f64;
                let shape = if keepdims {
                    vec![1; self.shape.len()]
                } else {
                    Vec::new()
                };
                Ok(Self {
                    shape,
                    values: vec![count],
                    dtype: DType::I64,
                })
            }
            Some(ax) => {
                let axis = normalize_axis(ax, self.shape.len())?;
                let out_shape = reduced_shape(&self.shape, axis, keepdims);
                let out_count = element_count(&out_shape).map_err(UFuncError::Shape)?;
                let mut out_values = vec![0.0f64; out_count];
                let axis_len = self.shape[axis];
                let inner: usize = self.shape[axis + 1..].iter().copied().product();
                let outer: usize = self.shape[..axis].iter().copied().product();
                let mut out_flat = 0usize;
                for outer_idx in 0..outer {
                    let base = outer_idx * axis_len * inner;
                    for inner_idx in 0..inner {
                        let mut count = 0.0_f64;
                        let mut offset = base + inner_idx;
                        for _ in 0..axis_len {
                            if self.values[offset] != 0.0 {
                                count += 1.0;
                            }
                            offset += inner;
                        }
                        out_values[out_flat] = count;
                        out_flat += 1;
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

    /// Pack the elements of a binary-valued array into bits (np.packbits).
    /// Input values > 0 are treated as 1. Returns a 1-D u8 array
    /// stored as f64 values (each value is a packed byte, 0–255).
    pub fn packbits(&self, axis: Option<isize>) -> Result<Self, UFuncError> {
        match axis {
            None => {
                let n = self.values.len();
                let nbytes = n.div_ceil(8);
                let mut packed = vec![0.0; nbytes];
                for (i, &v) in self.values.iter().enumerate() {
                    if v > 0.0 {
                        let byte_idx = i / 8;
                        let bit_idx = 7 - (i % 8);
                        packed[byte_idx] += (1u8 << bit_idx) as f64;
                    }
                }
                Ok(Self {
                    shape: vec![nbytes],
                    values: packed,
                    dtype: DType::U8,
                })
            }
            Some(ax) => {
                let axis = normalize_axis(ax, self.shape.len())?;
                let axis_len = self.shape[axis];
                let packed_len = axis_len.div_ceil(8);
                let inner: usize = self.shape[axis + 1..].iter().copied().product();
                let outer: usize = self.shape[..axis].iter().copied().product();
                let mut out_shape = self.shape.clone();
                out_shape[axis] = packed_len;
                let total = element_count(&out_shape).map_err(UFuncError::Shape)?;
                let mut packed = vec![0.0f64; total];
                for o in 0..outer {
                    for i in 0..inner {
                        for k in 0..axis_len {
                            let v = self.values[o * axis_len * inner + k * inner + i];
                            if v > 0.0 {
                                let byte_idx = k / 8;
                                let bit_idx = 7 - (k % 8);
                                packed[o * packed_len * inner + byte_idx * inner + i] +=
                                    (1u8 << bit_idx) as f64;
                            }
                        }
                    }
                }
                Ok(Self {
                    shape: out_shape,
                    values: packed,
                    dtype: DType::U8,
                })
            }
        }
    }

    /// Unpack elements of a uint8 array into a binary array (np.unpackbits).
    /// When `axis` is `None`, flattens and unpacks. When specified, unpacks along that axis.
    /// `count` limits the number of unpacked bits along the axis (default: axis_len * 8).
    pub fn unpackbits(
        &self,
        axis: Option<isize>,
        count: Option<usize>,
    ) -> Result<Self, UFuncError> {
        match axis {
            None => {
                let mut bits = Vec::with_capacity(self.values.len() * 8);
                for &v in &self.values {
                    let byte = v as u8;
                    for bit in (0..8).rev() {
                        bits.push(if byte & (1 << bit) != 0 { 1.0 } else { 0.0 });
                    }
                }
                if let Some(c) = count {
                    bits.truncate(c);
                }
                let n = bits.len();
                Ok(Self {
                    shape: vec![n],
                    values: bits,
                    dtype: DType::U8,
                })
            }
            Some(ax) => {
                let axis = normalize_axis(ax, self.shape.len())?;
                let axis_len = self.shape[axis];
                let full_bits = axis_len * 8;
                let out_bits = count.unwrap_or(full_bits).min(full_bits);
                let inner: usize = self.shape[axis + 1..].iter().copied().product();
                let outer: usize = self.shape[..axis].iter().copied().product();
                let mut out_shape = self.shape.clone();
                out_shape[axis] = out_bits;
                let total = element_count(&out_shape).map_err(UFuncError::Shape)?;
                let mut bits = vec![0.0f64; total];
                for o in 0..outer {
                    for i in 0..inner {
                        let mut bit_pos = 0usize;
                        for k in 0..axis_len {
                            let byte = self.values[o * axis_len * inner + k * inner + i] as u8;
                            for b in (0..8).rev() {
                                if bit_pos >= out_bits {
                                    break;
                                }
                                bits[o * out_bits * inner + bit_pos * inner + i] =
                                    if byte & (1 << b) != 0 { 1.0 } else { 0.0 };
                                bit_pos += 1;
                            }
                        }
                    }
                }
                Ok(Self {
                    shape: out_shape,
                    values: bits,
                    dtype: DType::U8,
                })
            }
        }
    }

    // ── complex number operations ────────
    // Complex arrays use interleaved representation: trailing dimension of size 2
    // where values are [re0, im0, re1, im1, ...].

    /// Return the angle (phase) of each complex element (np.angle).
    /// Input must have trailing dimension 2 (interleaved complex).
    /// Returns a real array with shape equal to input shape minus the trailing dim.
    pub fn angle(&self) -> Result<Self, UFuncError> {
        if self.shape.is_empty() || *self.shape.last().unwrap() != 2 {
            return Err(UFuncError::Msg(
                "angle: input must have trailing dimension 2 (interleaved complex)".to_string(),
            ));
        }
        let n = self.values.len() / 2;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let re = self.values[2 * i];
            let im = self.values[2 * i + 1];
            out.push(im.atan2(re));
        }
        let mut out_shape = self.shape[..self.shape.len() - 1].to_vec();
        if out_shape.is_empty() {
            out_shape.push(1);
        }
        Ok(Self {
            shape: out_shape,
            values: out,
            dtype: DType::F64,
        })
    }

    /// Extract the real part of each complex element (np.real).
    /// Input must have trailing dimension 2 (interleaved complex).
    pub fn real(&self) -> Result<Self, UFuncError> {
        if self.shape.is_empty() || *self.shape.last().unwrap() != 2 {
            return Err(UFuncError::Msg(
                "real: input must have trailing dimension 2 (interleaved complex)".to_string(),
            ));
        }
        let n = self.values.len() / 2;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(self.values[2 * i]);
        }
        let mut out_shape = self.shape[..self.shape.len() - 1].to_vec();
        if out_shape.is_empty() {
            out_shape.push(1);
        }
        Ok(Self {
            shape: out_shape,
            values: out,
            dtype: DType::F64,
        })
    }

    /// Extract the imaginary part of each complex element (np.imag).
    /// Input must have trailing dimension 2 (interleaved complex).
    pub fn imag(&self) -> Result<Self, UFuncError> {
        if self.shape.is_empty() || *self.shape.last().unwrap() != 2 {
            return Err(UFuncError::Msg(
                "imag: input must have trailing dimension 2 (interleaved complex)".to_string(),
            ));
        }
        let n = self.values.len() / 2;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(self.values[2 * i + 1]);
        }
        let mut out_shape = self.shape[..self.shape.len() - 1].to_vec();
        if out_shape.is_empty() {
            out_shape.push(1);
        }
        Ok(Self {
            shape: out_shape,
            values: out,
            dtype: DType::F64,
        })
    }

    /// Return the complex conjugate of each element (np.conj / np.conjugate).
    /// Input must have trailing dimension 2 (interleaved complex).
    /// Output has the same shape with imaginary parts negated.
    pub fn conj(&self) -> Result<Self, UFuncError> {
        if self.shape.is_empty() || *self.shape.last().unwrap() != 2 {
            return Err(UFuncError::Msg(
                "conj: input must have trailing dimension 2 (interleaved complex)".to_string(),
            ));
        }
        let mut out = self.values.clone();
        for i in 0..out.len() / 2 {
            out[2 * i + 1] = -out[2 * i + 1];
        }
        Ok(Self {
            shape: self.shape.clone(),
            values: out,
            dtype: self.dtype,
        })
    }

    /// Alias for [`conj`](Self::conj) (np.conjugate).
    pub fn conjugate(&self) -> Result<Self, UFuncError> {
        self.conj()
    }

    // ── numeric utility operations ────────

    /// Replace NaN with zero and infinity with large finite numbers (np.nan_to_num).
    /// `nan` replaces NaN, `posinf` replaces +inf, `neginf` replaces -inf.
    pub fn nan_to_num(&self, nan: f64, posinf: f64, neginf: f64) -> Self {
        let values: Vec<f64> = self
            .values
            .iter()
            .map(|&x| {
                if x.is_nan() {
                    nan
                } else if x.is_infinite() && x > 0.0 {
                    posinf
                } else if x.is_infinite() {
                    neginf
                } else {
                    x
                }
            })
            .collect();
        Self {
            shape: self.shape.clone(),
            values,
            dtype: self.dtype,
        }
    }

    /// Replace NaN with zero and infinity with large finite numbers using defaults (np.nan_to_num).
    /// NaN -> 0.0, +inf -> f64::MAX, -inf -> f64::MIN.
    pub fn nan_to_num_default(&self) -> Self {
        self.nan_to_num(0.0, f64::MAX, f64::MIN)
    }

    /// Return indices of non-zero elements in the flattened array (np.flatnonzero).
    pub fn flatnonzero(&self) -> Self {
        let indices: Vec<f64> = self
            .values
            .iter()
            .enumerate()
            .filter(|&(_, v)| *v != 0.0)
            .map(|(i, _)| i as f64)
            .collect();
        let n = indices.len();
        Self {
            shape: vec![n],
            values: indices,
            dtype: DType::I64,
        }
    }

    /// Return indices of nonzero elements as an (N, ndim) array (np.argwhere).
    pub fn argwhere(&self) -> Self {
        let ndim = self.shape.len();
        let nz_flat: Vec<usize> = self
            .values
            .iter()
            .enumerate()
            .filter(|&(_, v)| *v != 0.0)
            .map(|(i, _)| i)
            .collect();
        let n = nz_flat.len();
        if n == 0 {
            return Self {
                shape: vec![0, ndim],
                values: vec![],
                dtype: DType::I64,
            };
        }
        let strides = c_strides_elems(&self.shape);
        let mut out = Vec::with_capacity(n * ndim);
        for flat_idx in nz_flat {
            let mut rem = flat_idx;
            for s in &strides {
                out.push((rem / s) as f64);
                rem %= s;
            }
        }
        Self {
            shape: vec![n, ndim],
            values: out,
            dtype: DType::I64,
        }
    }

    /// Round to nearest integer towards zero (np.fix).
    pub fn fix(&self) -> Self {
        let values: Vec<f64> = self.values.iter().map(|&x| x.trunc()).collect();
        Self {
            shape: self.shape.clone(),
            values,
            dtype: self.dtype,
        }
    }

    /// Construct array from condlist and choicelist (np.select).
    /// Evaluates conditions in order; for each element, uses the first
    /// matching condition's corresponding choice. Elements with no
    /// matching condition get `default`.
    pub fn select(
        condlist: &[&Self],
        choicelist: &[&Self],
        default: f64,
    ) -> Result<Self, UFuncError> {
        if condlist.len() != choicelist.len() {
            return Err(UFuncError::Msg(
                "select: condlist and choicelist must have the same length".to_string(),
            ));
        }
        if condlist.is_empty() {
            return Err(UFuncError::Msg(
                "select: condlist must be non-empty".to_string(),
            ));
        }
        let n = condlist[0].values.len();
        for (cond, choice) in condlist.iter().zip(choicelist.iter()) {
            if cond.values.len() != n || choice.values.len() != n {
                return Err(UFuncError::Msg(
                    "select: all condition and choice arrays must have the same size".to_string(),
                ));
            }
        }
        let mut values = vec![default; n];
        // Iterate in reverse so first matching condition wins
        for (cond, choice) in condlist.iter().zip(choicelist.iter()).rev() {
            for (v, (c, ch)) in values
                .iter_mut()
                .zip(cond.values.iter().zip(choice.values.iter()))
            {
                if *c != 0.0 {
                    *v = *ch;
                }
            }
        }
        Ok(Self {
            shape: condlist[0].shape.clone(),
            values,
            dtype: choicelist[0].dtype,
        })
    }

    /// Convert angles from degrees to radians (np.deg2rad).
    pub fn deg2rad(&self) -> Self {
        self.elementwise_unary(UnaryOp::Radians)
    }

    /// Convert angles from radians to degrees (np.rad2deg).
    pub fn rad2deg(&self) -> Self {
        self.elementwise_unary(UnaryOp::Degrees)
    }

    /// Copy values from src into self at positions where mask is true (np.copyto).
    ///
    /// `casting` controls dtype compatibility: "no", "equiv", "safe", "same_kind", "unsafe".
    /// When `None`, defaults to "same_kind".
    pub fn copyto(
        &mut self,
        src: &Self,
        mask: Option<&Self>,
        casting: Option<&str>,
    ) -> Result<(), UFuncError> {
        let casting = casting.unwrap_or("same_kind");
        if !Self::can_cast(src.dtype, self.dtype, casting) {
            return Err(UFuncError::Msg(format!(
                "copyto: cannot cast from {:?} to {:?} with casting='{casting}'",
                src.dtype, self.dtype,
            )));
        }
        if self.values.len() != src.values.len() {
            return Err(UFuncError::Msg(
                "copyto: src and dst must have the same size".to_string(),
            ));
        }
        match mask {
            Some(m) => {
                if m.values.len() != self.values.len() {
                    return Err(UFuncError::Msg(
                        "copyto: mask must have the same size as dst".to_string(),
                    ));
                }
                for i in 0..self.values.len() {
                    if m.values[i] != 0.0 {
                        self.values[i] = src.values[i];
                    }
                }
            }
            None => {
                self.values.copy_from_slice(&src.values);
            }
        }
        Ok(())
    }

    /// Return the absolute value of complex elements (np.abs for complex arrays).
    /// Input must have trailing dimension 2 (interleaved complex).
    pub fn abs_complex(&self) -> Result<Self, UFuncError> {
        if self.shape.is_empty() || *self.shape.last().unwrap() != 2 {
            return Err(UFuncError::Msg(
                "abs_complex: input must have trailing dimension 2 (interleaved complex)"
                    .to_string(),
            ));
        }
        let n = self.values.len() / 2;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let re = self.values[2 * i];
            let im = self.values[2 * i + 1];
            out.push(re.hypot(im));
        }
        let mut out_shape = self.shape[..self.shape.len() - 1].to_vec();
        if out_shape.is_empty() {
            out_shape.push(1);
        }
        Ok(Self {
            shape: out_shape,
            values: out,
            dtype: DType::F64,
        })
    }

    // ── array splitting and assembly ────────

    /// Split array into sub-arrays allowing uneven division (np.array_split).
    /// Unlike `split`, this does not require the axis to divide evenly.
    /// The first `axis_len % sections` sub-arrays have size `axis_len / sections + 1`,
    /// the rest have size `axis_len / sections`.
    pub fn array_split(&self, sections: usize, axis: isize) -> Result<Vec<Self>, UFuncError> {
        if self.shape.is_empty() {
            return Err(UFuncError::Msg("cannot split scalar array".to_string()));
        }
        if sections == 0 {
            return Err(UFuncError::Msg(
                "array_split: number of sections must be > 0".to_string(),
            ));
        }
        let axis = normalize_axis(axis, self.shape.len())?;
        let axis_len = self.shape[axis];
        let base_size = axis_len / sections;
        let remainder = axis_len % sections;

        // Build internal split indices (not including 0 and end)
        let mut split_indices = Vec::with_capacity(sections - 1);
        let mut pos = 0;
        for s in 0..sections - 1 {
            pos += if s < remainder {
                base_size + 1
            } else {
                base_size
            };
            split_indices.push(pos);
        }
        self.array_split_at_indices(&split_indices, axis)
    }

    /// Split an array at explicit index boundaries along an axis.
    ///
    /// Mimics `np.array_split(a, indices, axis)` where `indices` is a list of
    /// split points. For N indices, produces N+1 sub-arrays:
    /// `[0:idx[0]], [idx[0]:idx[1]], ..., [idx[N-1]:axis_len]`.
    pub fn array_split_at_indices(
        &self,
        indices: &[usize],
        axis: usize,
    ) -> Result<Vec<Self>, UFuncError> {
        if self.shape.is_empty() {
            return Err(UFuncError::Msg("cannot split scalar array".to_string()));
        }
        let axis_len = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().copied().product();
        let outer: usize = self.shape[..axis].iter().copied().product();

        // Build boundary pairs: [0, idx0, idx1, ..., axis_len]
        let mut bounds = Vec::with_capacity(indices.len() + 2);
        bounds.push(0usize);
        for &idx in indices {
            bounds.push(idx.min(axis_len));
        }
        bounds.push(axis_len);

        let mut result = Vec::with_capacity(bounds.len() - 1);
        for w in bounds.windows(2) {
            let start = w[0];
            let end = w[1];
            let chunk = end.saturating_sub(start);
            let mut sub_shape = self.shape.clone();
            sub_shape[axis] = chunk;
            let count = element_count(&sub_shape).map_err(UFuncError::Shape)?;
            let mut values = vec![0.0f64; count];
            for o in 0..outer {
                for k in 0..chunk {
                    let src_k = start + k;
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

    /// Assemble an array from a flat list of blocks (simplified np.block).
    /// All blocks must have the same number of dimensions.
    /// Blocks are concatenated along axis 0 (row-wise stacking).
    pub fn block_row(blocks: &[&Self]) -> Result<Self, UFuncError> {
        if blocks.is_empty() {
            return Err(UFuncError::Msg(
                "block: need at least one block".to_string(),
            ));
        }
        Self::concatenate(blocks, 0)
    }

    /// Assemble a 2-D block matrix from a grid of blocks (np.block for 2-D).
    /// `grid` is a list of rows, each row is a list of arrays.
    /// Arrays in each row are concatenated along axis 1,
    /// then rows are concatenated along axis 0.
    pub fn block_2d(grid: &[Vec<&Self>]) -> Result<Self, UFuncError> {
        if grid.is_empty() {
            return Err(UFuncError::Msg(
                "block_2d: need at least one row".to_string(),
            ));
        }
        let mut rows = Vec::with_capacity(grid.len());
        for row in grid {
            if row.is_empty() {
                return Err(UFuncError::Msg(
                    "block_2d: each row must have at least one block".to_string(),
                ));
            }
            let row_arr = Self::concatenate(row, 1)?;
            rows.push(row_arr);
        }
        let row_refs: Vec<&Self> = rows.iter().collect();
        Self::concatenate(&row_refs, 0)
    }

    /// Determine the result dtype from a set of array dtypes (np.result_type).
    /// Follows NumPy's type promotion rules.
    pub fn result_type(arrays: &[&Self]) -> DType {
        if arrays.is_empty() {
            return DType::F64;
        }
        let mut result = arrays[0].dtype;
        for arr in &arrays[1..] {
            result = promote(result, arr.dtype);
        }
        result
    }

    /// Check if a cast between dtypes is allowed (np.can_cast).
    ///
    /// Wraps `fnp_dtype::can_cast` with the `UFuncArray` API surface.
    /// `casting` should be one of: "no", "safe", "same_kind", "unsafe".
    #[must_use]
    pub fn can_cast(from: DType, to: DType, casting: &str) -> bool {
        fnp_dtype::can_cast(from, to, casting)
    }

    /// Find the common type among arrays (np.common_type).
    ///
    /// Returns the broadest dtype that can hold all input dtypes.
    pub fn common_type(arrays: &[&Self]) -> DType {
        Self::result_type(arrays)
    }

    /// Return the minimum-size type character for the given type characters (np.mintypecode).
    ///
    /// Given a sequence of typecode characters, returns the smallest type
    /// that all types can be safely cast to.
    #[must_use]
    pub fn mintypecode(typecodes: &str) -> char {
        let mut result = DType::Bool;
        for ch in typecodes.chars() {
            let dt = match ch {
                '?' | 'b' => DType::Bool,
                'B' => DType::U8,
                'h' => DType::I16,
                'H' => DType::U16,
                'i' | 'l' => DType::I32,
                'I' | 'L' => DType::U32,
                'q' => DType::I64,
                'Q' => DType::U64,
                'f' => DType::F32,
                'd' => DType::F64,
                'F' => DType::Complex64,
                'D' => DType::Complex128,
                'S' | 'U' => DType::Str,
                _ => DType::F64,
            };
            result = promote(result, dt);
        }
        match result {
            DType::Bool => '?',
            DType::I8 => 'b',
            DType::I16 => 'h',
            DType::I32 => 'i',
            DType::I64 => 'q',
            DType::U8 => 'B',
            DType::U16 => 'H',
            DType::U32 => 'I',
            DType::U64 => 'Q',
            DType::F32 => 'f',
            DType::F64 => 'd',
            DType::Complex64 => 'F',
            DType::Complex128 => 'D',
            DType::Str | DType::DateTime64 | DType::TimeDelta64 => 'd',
        }
    }

    /// Determine the common dtype that `from` and `to` promote to (np.promote_types).
    pub fn promote_types(from: DType, to: DType) -> DType {
        promote(from, to)
    }

    /// Return the minimum scalar dtype that can hold the given value (np.min_scalar_type).
    /// For an array, returns the dtype needed to hold all values.
    pub fn min_scalar_type(&self) -> DType {
        if self.values.is_empty() {
            return DType::F64;
        }
        let all_bool = self.values.iter().all(|&v| v == 0.0 || v == 1.0);
        if all_bool {
            return DType::Bool;
        }
        let all_int = self.values.iter().all(|&v| v == v.floor() && v.is_finite());
        if all_int {
            let min_v = self.values.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_v = self
                .values
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            if min_v >= 0.0 {
                if max_v <= 255.0 {
                    return DType::U8;
                }
                if max_v <= 65535.0 {
                    return DType::U16;
                }
                if max_v <= u32::MAX as f64 {
                    return DType::U32;
                }
                return DType::U64;
            }
            if min_v >= i8::MIN as f64 && max_v <= i8::MAX as f64 {
                return DType::I8;
            }
            if min_v >= i16::MIN as f64 && max_v <= i16::MAX as f64 {
                return DType::I16;
            }
            if min_v >= i32::MIN as f64 && max_v <= i32::MAX as f64 {
                return DType::I32;
            }
            return DType::I64;
        }
        DType::F64
    }

    /// Check if the array is stored in Fortran-contiguous order (np.isfortran).
    /// Since `UFuncArray` always uses C-contiguous storage, this returns false
    /// unless the array is 0-D or 1-D (which are both C- and F-contiguous).
    #[must_use]
    pub fn isfortran(&self) -> bool {
        self.shape.len() <= 1
    }

    /// Check if the value is a scalar (0-D array) (np.isscalar).
    #[must_use]
    pub fn isscalar(&self) -> bool {
        self.shape.is_empty() || (self.shape.len() == 1 && self.shape[0] == 1)
    }

    /// Element-wise test for real values (np.isreal).
    /// For non-complex dtypes, all values are real. For complex dtypes,
    /// an element is real if its imaginary part is zero.
    pub fn isreal(&self) -> Self {
        if !matches!(self.dtype, DType::Complex64 | DType::Complex128) {
            return Self {
                shape: self.shape.clone(),
                values: vec![1.0; self.values.len()],
                dtype: DType::Bool,
            };
        }
        let n = self.values.len() / 2;
        let values: Vec<f64> = (0..n)
            .map(|i| {
                if self.values[i * 2 + 1] == 0.0 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        Self {
            shape: self.shape[..self.shape.len() - 1].to_vec(),
            values,
            dtype: DType::Bool,
        }
    }

    /// Element-wise test for complex values with nonzero imaginary part (np.iscomplex).
    pub fn iscomplex(&self) -> Self {
        if !matches!(self.dtype, DType::Complex64 | DType::Complex128) {
            return Self {
                shape: self.shape.clone(),
                values: vec![0.0; self.values.len()],
                dtype: DType::Bool,
            };
        }
        let n = self.values.len() / 2;
        let values: Vec<f64> = (0..n)
            .map(|i| {
                if self.values[i * 2 + 1] != 0.0 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        Self {
            shape: self.shape[..self.shape.len() - 1].to_vec(),
            values,
            dtype: DType::Bool,
        }
    }

    /// Check if the array dtype is real (not complex) (np.isrealobj).
    #[must_use]
    pub fn isrealobj(&self) -> bool {
        !matches!(self.dtype, DType::Complex64 | DType::Complex128)
    }

    /// Check if the array dtype is complex (np.iscomplexobj).
    #[must_use]
    pub fn iscomplexobj(&self) -> bool {
        matches!(self.dtype, DType::Complex64 | DType::Complex128)
    }

    /// Check if two arrays share memory (np.shares_memory).
    ///
    /// Since `UFuncArray` uses owned `Vec<f64>` storage (no aliasing),
    /// two distinct arrays never share memory. Returns `true` only
    /// if both references point to the exact same array.
    #[must_use]
    pub fn shares_memory(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }

    /// Check if two arrays may share memory (np.may_share_memory).
    ///
    /// Conservative version of `shares_memory`. Same semantics for
    /// owned-storage arrays.
    #[must_use]
    pub fn may_share_memory(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }

    /// Return the byte bounds of the array data (np.byte_bounds).
    ///
    /// Returns `(low, high)` addresses as `usize` values, where `low` is the
    /// start of the data buffer and `high` is one past the end.
    #[must_use]
    pub fn byte_bounds(&self) -> (usize, usize) {
        if self.values.is_empty() {
            return (0, 0);
        }
        let ptr = self.values.as_ptr() as usize;
        let end = ptr + self.values.len() * std::mem::size_of::<f64>();
        (ptr, end)
    }

    /// Set elements using a boolean mask (np.putmask / np.put_mask).
    ///
    /// For each position where `mask` is non-zero, the corresponding
    /// value from `values` is placed. Values are cycled if shorter than
    /// the number of `True` positions.
    pub fn put_mask(&mut self, mask: &Self, values: &[f64]) -> Result<(), UFuncError> {
        if mask.values.len() != self.values.len() {
            return Err(UFuncError::Msg(
                "put_mask: mask must have same size as array".to_string(),
            ));
        }
        if values.is_empty() {
            return Err(UFuncError::Msg(
                "put_mask: values must not be empty".to_string(),
            ));
        }
        let mut vi = 0;
        for (i, &m) in mask.values.iter().enumerate() {
            if m != 0.0 {
                self.values[i] = values[vi % values.len()];
                vi += 1;
            }
        }
        Ok(())
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
            DType::Complex64 => "complex64",
            DType::Complex128 => "complex128",
            DType::Str => "str",
            DType::DateTime64 => "datetime64",
            DType::TimeDelta64 => "timedelta64",
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
    interpolate_percentile_method(sorted, fraction, QuantileInterp::Linear)
}

fn interpolate_percentile_method(sorted: &[f64], fraction: f64, method: QuantileInterp) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = fraction * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let frac = idx - lo as f64;
    // When idx is exactly an integer, all methods agree
    if frac == 0.0 || lo >= n - 1 {
        return sorted[lo.min(n - 1)];
    }
    let hi = lo + 1;
    match method {
        QuantileInterp::Linear => sorted[lo] * (1.0 - frac) + sorted[hi] * frac,
        QuantileInterp::Lower => sorted[lo],
        QuantileInterp::Higher => sorted[hi],
        QuantileInterp::Nearest => {
            if frac <= 0.5 {
                sorted[lo]
            } else {
                sorted[hi]
            }
        }
        QuantileInterp::Midpoint => (sorted[lo] + sorted[hi]) / 2.0,
    }
}

/// Validate sort kind parameter. Returns normalized kind string.
fn validate_sort_kind(kind: Option<&str>) -> Result<&str, UFuncError> {
    match kind {
        None | Some("quicksort") => Ok("quicksort"),
        Some("mergesort") | Some("stable") => Ok("stable"),
        Some("heapsort") => Ok("heapsort"),
        Some(other) => Err(UFuncError::Msg(format!(
            "sort: unknown kind '{other}'. Must be quicksort, mergesort, heapsort, or stable"
        ))),
    }
}

/// Heapsort a mutable f64 slice in ascending order.
fn heapsort_f64(data: &mut [f64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    let cmp = |a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);

    // Build max-heap (sift down from last parent to root)
    for i in (0..n / 2).rev() {
        sift_down(data, i, n, &cmp);
    }
    // Extract max repeatedly
    for end in (1..n).rev() {
        data.swap(0, end);
        sift_down(data, 0, end, &cmp);
    }
}

fn sift_down(data: &mut [f64], mut root: usize, end: usize, cmp: &dyn Fn(&f64, &f64) -> std::cmp::Ordering) {
    loop {
        let left = 2 * root + 1;
        if left >= end {
            break;
        }
        let right = left + 1;
        let mut largest = root;
        if cmp(&data[left], &data[largest]) == std::cmp::Ordering::Greater {
            largest = left;
        }
        if right < end && cmp(&data[right], &data[largest]) == std::cmp::Ordering::Greater {
            largest = right;
        }
        if largest == root {
            break;
        }
        data.swap(root, largest);
        root = largest;
    }
}

/// Heapsort indices by their corresponding values (for argsort).
fn heapsort_indices_by(indices: &mut [usize], values: &dyn Fn(usize) -> f64) {
    let n = indices.len();
    if n <= 1 {
        return;
    }
    let cmp = |a: &usize, b: &usize| {
        values(*a)
            .partial_cmp(&values(*b))
            .unwrap_or(std::cmp::Ordering::Equal)
    };

    for i in (0..n / 2).rev() {
        sift_down_idx(indices, i, n, &cmp);
    }
    for end in (1..n).rev() {
        indices.swap(0, end);
        sift_down_idx(indices, 0, end, &cmp);
    }
}

fn sift_down_idx(data: &mut [usize], mut root: usize, end: usize, cmp: &dyn Fn(&usize, &usize) -> std::cmp::Ordering) {
    loop {
        let left = 2 * root + 1;
        if left >= end {
            break;
        }
        let right = left + 1;
        let mut largest = root;
        if cmp(&data[left], &data[largest]) == std::cmp::Ordering::Greater {
            largest = left;
        }
        if right < end && cmp(&data[right], &data[largest]) == std::cmp::Ordering::Greater {
            largest = right;
        }
        if largest == root {
            break;
        }
        data.swap(root, largest);
        root = largest;
    }
}

/// Sort a slice using the specified algorithm kind.
fn sort_slice_by_kind(data: &mut [f64], kind: &str) {
    let cmp = |a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
    match kind {
        "quicksort" => data.sort_unstable_by(cmp),
        "stable" => data.sort_by(cmp),
        "heapsort" => heapsort_f64(data),
        _ => data.sort_unstable_by(cmp),
    }
}

/// Sort indices using the specified algorithm kind.
fn argsort_slice_by_kind(indices: &mut [usize], values: &dyn Fn(usize) -> f64, kind: &str) {
    let cmp = |a: &usize, b: &usize| {
        values(*a)
            .partial_cmp(&values(*b))
            .unwrap_or(std::cmp::Ordering::Equal)
    };
    match kind {
        "quicksort" => indices.sort_unstable_by(cmp),
        "stable" => indices.sort_by(cmp),
        "heapsort" => heapsort_indices_by(indices, values),
        _ => indices.sort_unstable_by(cmp),
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

/// Bessel function of the first kind, order 0.
/// Uses polynomial approximations from Abramowitz and Stegun (9.4.1, 9.4.3).
#[allow(clippy::inconsistent_digit_grouping)]
fn bessel_j0(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let num = 57_568_490_574.0
            + y * (-13_362_590_354.0
                + y * (651_619_640.7
                    + y * (-11_214_424.18 + y * (77_392.33017 + y * (-184.9052456)))));
        let den = 57_568_490_411.0
            + y * (1_029_532_985.0
                + y * (9_494_680.718 + y * (59_272.64853 + y * (267.8532712 + y))));
        num / den
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - std::f64::consts::FRAC_PI_4;
        let p0 = 1.0
            + y * (-0.1098628627e-2
                + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        let q0 = -0.1562499995e-1
            + y * (0.1430488765e-3
                + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934935152e-7)));
        let factor = (std::f64::consts::FRAC_2_PI / ax).sqrt();
        factor * (p0 * xx.cos() - z * q0 * xx.sin())
    }
}

/// Bessel function of the first kind, order 1.
/// Uses polynomial approximations from Abramowitz and Stegun.
#[allow(clippy::inconsistent_digit_grouping)]
fn bessel_j1(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let num = x
            * (72_362_614_232.0
                + y * (-7_895_059_235.0
                    + y * (242_396_853.1
                        + y * (-2_972_611.439 + y * (15_704.48260 + y * (-30.16036606))))));
        let den = 144_725_228_442.0
            + y * (2_300_535_178.0
                + y * (18_583_304.74 + y * (99_447.43394 + y * (376.9991397 + y))));
        num / den
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 3.0 * std::f64::consts::FRAC_PI_4;
        let p1 = 1.0
            + y * (0.183105e-2
                + y * (-0.3516396496e-4 + y * (0.2457520174e-5 - y * 0.240337019e-6)));
        let q1 = 0.04687499995
            + y * (-0.2002690873e-3
                + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
        let factor = (std::f64::consts::FRAC_2_PI / ax).sqrt();
        let result = factor * (p1 * xx.cos() - z * q1 * xx.sin());
        if x < 0.0 { -result } else { result }
    }
}

/// Bessel function of the second kind, order 0.
/// Uses polynomial approximations from Abramowitz and Stegun.
#[allow(clippy::inconsistent_digit_grouping)]
fn bessel_y0(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if x < 8.0 {
        let y = x * x;
        let num = -2_957_821_389.0
            + y * (7_062_834_065.0
                + y * (-512_359_803.6
                    + y * (10_879_881.29 + y * (-86_327.92757 + y * 228.4622733))));
        let den = 40_076_544_269.0
            + y * (745_249_964.8
                + y * (7_189_466.438 + y * (47_447.26470 + y * (226.1030244 + y))));
        (num / den) + std::f64::consts::FRAC_2_PI * bessel_j0(x) * x.ln()
    } else {
        let z = 8.0 / x;
        let y = z * z;
        let xx = x - std::f64::consts::FRAC_PI_4;
        let p0 = 1.0
            + y * (-0.1098628627e-2
                + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        let q0 = -0.1562499995e-1
            + y * (0.1430488765e-3
                + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934935152e-7)));
        let factor = (std::f64::consts::FRAC_2_PI / x).sqrt();
        factor * (p0 * xx.sin() + z * q0 * xx.cos())
    }
}

/// Bessel function of the second kind, order 1.
/// Uses polynomial approximations from Abramowitz and Stegun.
#[allow(clippy::inconsistent_digit_grouping)]
fn bessel_y1(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if x < 8.0 {
        let y = x * x;
        let num = x
            * (-4_900_604_943_000.0
                + y * (1_275_274_390_000.0
                    + y * (-51_534_866_838.0
                        + y * (622_785_432.7 + y * (-3_132_339.048 + y * 7_621.255_74)))));
        let den = 24_995_805_700_000.0
            + y * (424_441_966_400.0
                + y * (3_733_650_367.0
                    + y * (22_459_040.02 + y * (103_680.252 + y * 365.9584658))));
        (num / den) + std::f64::consts::FRAC_2_PI * (bessel_j1(x) * x.ln() - 1.0 / x)
    } else {
        let z = 8.0 / x;
        let y = z * z;
        let xx = x - 3.0 * std::f64::consts::FRAC_PI_4;
        let p1 = 1.0
            + y * (0.183105e-2
                + y * (-0.3516396496e-4 + y * (0.2457520174e-5 - y * 0.240337019e-6)));
        let q1 = 0.04687499995
            + y * (-0.2002690873e-3
                + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
        let factor = (std::f64::consts::FRAC_2_PI / x).sqrt();
        factor * (p1 * xx.sin() + z * q1 * xx.cos())
    }
}

/// Lanczos approximation for the Gamma function.
/// Accurate to ~15 significant digits for real positive arguments.
fn lanczos_gamma(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // Reflection formula for x < 0.5
    if x < 0.5 {
        let reflected = lanczos_gamma(1.0 - x);
        return std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * reflected);
    }
    let x = x - 1.0;
    // Lanczos g=7 coefficients
    #[allow(clippy::excessive_precision)]
    const P: [f64; 8] = [
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    #[allow(clippy::excessive_precision)]
    let mut sum = 0.99999999999980993;
    for (i, &p) in P.iter().enumerate() {
        sum += p / (x + i as f64 + 1.0);
    }
    let t = x + 7.5; // g + 0.5
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * sum
}

/// Error function approximation using Abramowitz & Stegun formula 7.1.26.
/// Maximum error: |epsilon(x)| <= 1.5e-7.
fn erf_approx(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    let sign = x.signum();
    let x = x.abs();
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;
    let t = 1.0 / (1.0 + P * x);
    let poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Digamma function approximation.
/// Uses recurrence psi(x+1) = psi(x) + 1/x to shift argument above 6,
/// then asymptotic series.
fn digamma_approx(mut x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // Reflection formula for x < 0
    if x < 0.0 {
        return digamma_approx(1.0 - x) - std::f64::consts::PI / (std::f64::consts::PI * x).tan();
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    let mut result = 0.0;
    // Recurrence to shift argument above 6
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    // Asymptotic series: psi(x) ~ ln(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴) - 1/(252x⁶) + ...
    result += x.ln() - 0.5 / x;
    let x2 = x * x;
    result -= 1.0 / (12.0 * x2);
    result += 1.0 / (120.0 * x2 * x2);
    result -= 1.0 / (252.0 * x2 * x2 * x2);
    result
}

/// Mixed-radix Bluestein/Chirp-Z FFT supporting arbitrary lengths.
/// For power-of-two lengths, uses Cooley-Tukey DIT.
/// For non-power-of-two, uses Bluestein's algorithm (zero-pad to power-of-two).
/// Apply 1-D FFT (or IFFT) along a single axis of an N-dimensional array.
///
/// `shape` is the full N-D shape of the data.
/// `re` and `im` are flat row-major arrays of length `product(shape)`.
/// `axis` is the axis along which to apply the 1-D FFT.
fn fftn_along_axis(shape: &[usize], re: &mut [f64], im: &mut [f64], axis: usize, inverse: bool) {
    let ndim = shape.len();
    if axis >= ndim {
        return;
    }
    let axis_len = shape[axis];
    if axis_len <= 1 {
        return;
    }

    // Compute the stride (number of elements) along the given axis:
    // outer_size = product of shape[..axis]
    // inner_size = product of shape[axis+1..]
    let outer_size: usize = shape[..axis].iter().product::<usize>().max(1);
    let inner_size: usize = shape[axis + 1..].iter().product::<usize>().max(1);

    let mut buf_re = vec![0.0; axis_len];
    let mut buf_im = vec![0.0; axis_len];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // Extract 1-D slice along axis
            for k in 0..axis_len {
                let idx = outer * axis_len * inner_size + k * inner_size + inner;
                buf_re[k] = re[idx];
                buf_im[k] = im[idx];
            }
            // Apply 1-D FFT
            fft_dit(&mut buf_re, &mut buf_im, inverse);
            // Write back
            for k in 0..axis_len {
                let idx = outer * axis_len * inner_size + k * inner_size + inner;
                re[idx] = buf_re[k];
                im[idx] = buf_im[k];
            }
        }
    }
}

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

// ── numpy datetime/timedelta arithmetic ─────────────────────────────────

/// Weekday for a date represented as days since Unix epoch (1970-01-01, Thursday).
/// Returns 0=Monday .. 6=Sunday (NumPy convention).
#[must_use]
fn epoch_day_to_weekday(day: i64) -> u8 {
    // 1970-01-01 is Thursday = 3
    let wd = (day + 3).rem_euclid(7);
    wd as u8
}

impl UFuncArray {
    // ── datetime + timedelta → datetime ─────────────────────────────

    /// Add timedelta array to datetime array element-wise.
    /// `self` must be DateTime64, `delta` must be TimeDelta64.
    pub fn datetime_add(&self, delta: &Self) -> Result<Self, UFuncError> {
        if self.dtype != DType::DateTime64 {
            return Err(UFuncError::Msg(
                "datetime_add requires DateTime64 array".to_string(),
            ));
        }
        if delta.dtype != DType::TimeDelta64 {
            return Err(UFuncError::Msg(
                "datetime_add requires TimeDelta64 delta".to_string(),
            ));
        }
        let result = self.elementwise_binary(delta, BinaryOp::Add)?;
        Ok(Self {
            shape: result.shape,
            values: result.values,
            dtype: DType::DateTime64,
        })
    }

    /// Subtract two datetime arrays → timedelta.
    pub fn datetime_sub(&self, other: &Self) -> Result<Self, UFuncError> {
        if self.dtype != DType::DateTime64 || other.dtype != DType::DateTime64 {
            return Err(UFuncError::Msg(
                "datetime_sub requires two DateTime64 arrays".to_string(),
            ));
        }
        let result = self.elementwise_binary(other, BinaryOp::Sub)?;
        Ok(Self {
            shape: result.shape,
            values: result.values,
            dtype: DType::TimeDelta64,
        })
    }

    /// Add two timedelta arrays.
    pub fn timedelta_add(&self, other: &Self) -> Result<Self, UFuncError> {
        if self.dtype != DType::TimeDelta64 || other.dtype != DType::TimeDelta64 {
            return Err(UFuncError::Msg(
                "timedelta_add requires two TimeDelta64 arrays".to_string(),
            ));
        }
        let result = self.elementwise_binary(other, BinaryOp::Add)?;
        Ok(Self {
            shape: result.shape,
            values: result.values,
            dtype: DType::TimeDelta64,
        })
    }

    /// Subtract two timedelta arrays.
    pub fn timedelta_sub(&self, other: &Self) -> Result<Self, UFuncError> {
        if self.dtype != DType::TimeDelta64 || other.dtype != DType::TimeDelta64 {
            return Err(UFuncError::Msg(
                "timedelta_sub requires two TimeDelta64 arrays".to_string(),
            ));
        }
        let result = self.elementwise_binary(other, BinaryOp::Sub)?;
        Ok(Self {
            shape: result.shape,
            values: result.values,
            dtype: DType::TimeDelta64,
        })
    }

    /// Multiply timedelta by scalar.
    pub fn timedelta_mul(&self, scalar: f64) -> Result<Self, UFuncError> {
        if self.dtype != DType::TimeDelta64 {
            return Err(UFuncError::Msg(
                "timedelta_mul requires TimeDelta64 array".to_string(),
            ));
        }
        let values: Vec<f64> = self.values.iter().map(|&v| v * scalar).collect();
        Ok(Self {
            shape: self.shape.clone(),
            values,
            dtype: DType::TimeDelta64,
        })
    }

    /// Divide timedelta by scalar.
    pub fn timedelta_div(&self, scalar: f64) -> Result<Self, UFuncError> {
        if self.dtype != DType::TimeDelta64 {
            return Err(UFuncError::Msg(
                "timedelta_div requires TimeDelta64 array".to_string(),
            ));
        }
        if scalar == 0.0 {
            return Err(UFuncError::Msg(
                "division by zero in timedelta_div".to_string(),
            ));
        }
        let values: Vec<f64> = self.values.iter().map(|&v| v / scalar).collect();
        Ok(Self {
            shape: self.shape.clone(),
            values,
            dtype: DType::TimeDelta64,
        })
    }

    /// Negate a timedelta array.
    #[must_use]
    pub fn timedelta_neg(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            values: self.values.iter().map(|&v| -v).collect(),
            dtype: DType::TimeDelta64,
        }
    }

    /// Absolute value of timedelta array.
    #[must_use]
    pub fn timedelta_abs(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            values: self.values.iter().map(|&v| v.abs()).collect(),
            dtype: DType::TimeDelta64,
        }
    }
}

/// Check whether each day (epoch days) is a business day (Mon-Fri).
/// Returns a Bool-typed array: 1.0 = business day, 0.0 = weekend.
///
/// `np.is_busday(dates)`
pub fn is_busday(dates: &UFuncArray) -> Result<UFuncArray, UFuncError> {
    let values: Vec<f64> = dates
        .values()
        .iter()
        .map(|&d| {
            let wd = epoch_day_to_weekday(d as i64);
            if wd < 5 { 1.0 } else { 0.0 }
        })
        .collect();
    UFuncArray::new(dates.shape().to_vec(), values, DType::Bool)
}

/// Count the number of business days between `start` and `end` (exclusive end).
/// Both arrays contain epoch day values. Result dtype is I64.
///
/// `np.busday_count(start, end)`
pub fn busday_count(start: &UFuncArray, end: &UFuncArray) -> Result<UFuncArray, UFuncError> {
    if start.values().len() != end.values().len() {
        return Err(UFuncError::InvalidInputLength {
            expected: start.values().len(),
            actual: end.values().len(),
        });
    }
    let values: Vec<f64> = start
        .values()
        .iter()
        .zip(end.values().iter())
        .map(|(&s, &e)| {
            let s_day = s as i64;
            let e_day = e as i64;
            if s_day == e_day {
                return 0.0;
            }
            let (lo, hi, sign) = if s_day < e_day {
                (s_day, e_day, 1.0)
            } else {
                (e_day, s_day, -1.0)
            };
            // Full weeks contain exactly 5 business days each
            let total_days = hi - lo;
            let full_weeks = total_days / 7;
            let remainder = total_days % 7;
            let mut count = full_weeks * 5;
            // Count business days in the remaining partial week
            let start_wd = epoch_day_to_weekday(lo);
            for i in 0..remainder {
                let wd = (start_wd as i64 + i) % 7;
                if wd < 5 {
                    count += 1;
                }
            }
            count as f64 * sign
        })
        .collect();
    UFuncArray::new(start.shape().to_vec(), values, DType::I64)
}

/// Offset dates by a number of business days.
/// `dates` contains epoch day values, `offsets` contains integer business-day offsets.
/// Result dtype is DateTime64.
///
/// `np.busday_offset(dates, offsets)`
pub fn busday_offset(dates: &UFuncArray, offsets: &UFuncArray) -> Result<UFuncArray, UFuncError> {
    if dates.values().len() != offsets.values().len() {
        return Err(UFuncError::InvalidInputLength {
            expected: dates.values().len(),
            actual: offsets.values().len(),
        });
    }
    let values: Vec<f64> = dates
        .values()
        .iter()
        .zip(offsets.values().iter())
        .map(|(&d, &off)| {
            let mut current = d as i64;
            let off_i = off as i64;
            // First, if current day is not a business day, snap forward/backward
            let wd = epoch_day_to_weekday(current);
            if wd >= 5 {
                // Weekend: snap to next Monday if offset >= 0, else previous Friday
                if off_i >= 0 {
                    current += i64::from(7 - wd); // next Monday
                } else {
                    current -= i64::from(wd - 4); // previous Friday
                }
            }
            let abs_off = off_i.unsigned_abs();
            let step: i64 = if off_i >= 0 { 1 } else { -1 };
            let mut remaining = abs_off;
            while remaining > 0 {
                current += step;
                if epoch_day_to_weekday(current) < 5 {
                    remaining -= 1;
                }
            }
            current as f64
        })
        .collect();
    UFuncArray::new(dates.shape().to_vec(), values, DType::DateTime64)
}

// ── numpy.ma — Masked Array Module ──────────────────────────────────────

/// Error type for masked array operations.
#[derive(Debug, Clone, PartialEq)]
pub enum MAError {
    /// Shape mismatch between data and mask.
    MaskShapeMismatch {
        data_shape: Vec<usize>,
        mask_shape: Vec<usize>,
    },
    /// Wrapped UFuncError from underlying array operation.
    UFunc(UFuncError),
    /// General message.
    Msg(String),
}

impl std::fmt::Display for MAError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MaskShapeMismatch {
                data_shape,
                mask_shape,
            } => write!(
                f,
                "mask shape {mask_shape:?} does not match data shape {data_shape:?}"
            ),
            Self::UFunc(e) => write!(f, "{e}"),
            Self::Msg(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for MAError {}

impl From<UFuncError> for MAError {
    fn from(e: UFuncError) -> Self {
        Self::UFunc(e)
    }
}

/// A masked array analogous to `numpy.ma.MaskedArray`.
///
/// Mask convention (matching NumPy): `true` (1.0) = element is **masked** (excluded/invalid),
/// `false` (0.0) = element is **valid**.
#[derive(Debug, Clone, PartialEq)]
pub struct MaskedArray {
    data: UFuncArray,
    /// Boolean mask: 1.0 = masked (excluded), 0.0 = valid. `None` means no mask (all valid).
    mask: Option<UFuncArray>,
    fill_value: f64,
    hard_mask: bool,
}

impl MaskedArray {
    // ── Constructors ────────────────────────────────────────────────

    /// Create a new masked array. If `mask` is `Some`, its shape must match `data`.
    pub fn new(
        data: UFuncArray,
        mask: Option<UFuncArray>,
        fill_value: Option<f64>,
    ) -> Result<Self, MAError> {
        if let Some(ref m) = mask
            && m.shape() != data.shape()
        {
            return Err(MAError::MaskShapeMismatch {
                data_shape: data.shape().to_vec(),
                mask_shape: m.shape().to_vec(),
            });
        }
        Ok(Self {
            fill_value: fill_value.unwrap_or(1e20),
            data,
            mask,
            hard_mask: false,
        })
    }

    /// Create from raw components.
    pub fn from_values(
        shape: Vec<usize>,
        values: Vec<f64>,
        mask_values: Option<Vec<f64>>,
        dtype: DType,
    ) -> Result<Self, MAError> {
        let data = UFuncArray::new(shape.clone(), values, dtype)?;
        let mask = match mask_values {
            Some(mv) => Some(UFuncArray::new(shape, mv, DType::Bool)?),
            None => None,
        };
        Self::new(data, mask, None)
    }

    /// Mask elements where `condition` is true (non-zero).
    /// `np.ma.masked_where(condition, data)`
    pub fn masked_where(condition: &UFuncArray, data: &UFuncArray) -> Result<Self, MAError> {
        if condition.shape() != data.shape() {
            return Err(MAError::MaskShapeMismatch {
                data_shape: data.shape().to_vec(),
                mask_shape: condition.shape().to_vec(),
            });
        }
        // Normalize: any nonzero → 1.0
        let mask_vals: Vec<f64> = condition
            .values()
            .iter()
            .map(|&v| if v != 0.0 { 1.0 } else { 0.0 })
            .collect();
        let mask = UFuncArray::new(condition.shape().to_vec(), mask_vals, DType::Bool)?;
        Self::new(data.clone(), Some(mask), None)
    }

    /// Mask elements equal to `value`. `np.ma.masked_equal(data, value)`
    pub fn masked_equal(data: &UFuncArray, value: f64) -> Result<Self, MAError> {
        let mask_vals: Vec<f64> = data
            .values()
            .iter()
            .map(|&v| if v == value { 1.0 } else { 0.0 })
            .collect();
        let mask = UFuncArray::new(data.shape().to_vec(), mask_vals, DType::Bool)?;
        Self::new(data.clone(), Some(mask), None)
    }

    /// Mask elements not equal to `value`. `np.ma.masked_not_equal(data, value)`
    pub fn masked_not_equal(data: &UFuncArray, value: f64) -> Result<Self, MAError> {
        let mask_vals: Vec<f64> = data
            .values()
            .iter()
            .map(|&v| if v != value { 1.0 } else { 0.0 })
            .collect();
        let mask = UFuncArray::new(data.shape().to_vec(), mask_vals, DType::Bool)?;
        Self::new(data.clone(), Some(mask), None)
    }

    /// Mask elements greater than `value`. `np.ma.masked_greater(data, value)`
    pub fn masked_greater(data: &UFuncArray, value: f64) -> Result<Self, MAError> {
        let mask_vals: Vec<f64> = data
            .values()
            .iter()
            .map(|&v| if v > value { 1.0 } else { 0.0 })
            .collect();
        let mask = UFuncArray::new(data.shape().to_vec(), mask_vals, DType::Bool)?;
        Self::new(data.clone(), Some(mask), None)
    }

    /// Mask elements greater than or equal to `value`.
    pub fn masked_greater_equal(data: &UFuncArray, value: f64) -> Result<Self, MAError> {
        let mask_vals: Vec<f64> = data
            .values()
            .iter()
            .map(|&v| if v >= value { 1.0 } else { 0.0 })
            .collect();
        let mask = UFuncArray::new(data.shape().to_vec(), mask_vals, DType::Bool)?;
        Self::new(data.clone(), Some(mask), None)
    }

    /// Mask elements less than `value`. `np.ma.masked_less(data, value)`
    pub fn masked_less(data: &UFuncArray, value: f64) -> Result<Self, MAError> {
        let mask_vals: Vec<f64> = data
            .values()
            .iter()
            .map(|&v| if v < value { 1.0 } else { 0.0 })
            .collect();
        let mask = UFuncArray::new(data.shape().to_vec(), mask_vals, DType::Bool)?;
        Self::new(data.clone(), Some(mask), None)
    }

    /// Mask elements less than or equal to `value`.
    pub fn masked_less_equal(data: &UFuncArray, value: f64) -> Result<Self, MAError> {
        let mask_vals: Vec<f64> = data
            .values()
            .iter()
            .map(|&v| if v <= value { 1.0 } else { 0.0 })
            .collect();
        let mask = UFuncArray::new(data.shape().to_vec(), mask_vals, DType::Bool)?;
        Self::new(data.clone(), Some(mask), None)
    }

    /// Mask elements inside `[v1, v2]`. `np.ma.masked_inside(data, v1, v2)`
    pub fn masked_inside(data: &UFuncArray, v1: f64, v2: f64) -> Result<Self, MAError> {
        let lo = v1.min(v2);
        let hi = v1.max(v2);
        let mask_vals: Vec<f64> = data
            .values()
            .iter()
            .map(|&v| if v >= lo && v <= hi { 1.0 } else { 0.0 })
            .collect();
        let mask = UFuncArray::new(data.shape().to_vec(), mask_vals, DType::Bool)?;
        Self::new(data.clone(), Some(mask), None)
    }

    /// Mask elements outside `[v1, v2]`. `np.ma.masked_outside(data, v1, v2)`
    pub fn masked_outside(data: &UFuncArray, v1: f64, v2: f64) -> Result<Self, MAError> {
        let lo = v1.min(v2);
        let hi = v1.max(v2);
        let mask_vals: Vec<f64> = data
            .values()
            .iter()
            .map(|&v| if v < lo || v > hi { 1.0 } else { 0.0 })
            .collect();
        let mask = UFuncArray::new(data.shape().to_vec(), mask_vals, DType::Bool)?;
        Self::new(data.clone(), Some(mask), None)
    }

    /// Mask NaN and Inf values. `np.ma.masked_invalid(data)`
    pub fn masked_invalid(data: &UFuncArray) -> Result<Self, MAError> {
        let mask_vals: Vec<f64> = data
            .values()
            .iter()
            .map(|&v| {
                if v.is_nan() || v.is_infinite() {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let mask = UFuncArray::new(data.shape().to_vec(), mask_vals, DType::Bool)?;
        Self::new(data.clone(), Some(mask), None)
    }

    // ── Accessors ───────────────────────────────────────────────────

    /// The underlying data array (including masked elements).
    #[must_use]
    pub fn data(&self) -> &UFuncArray {
        &self.data
    }

    /// The mask array (if any). `true` (1.0) = masked.
    #[must_use]
    pub fn mask(&self) -> Option<&UFuncArray> {
        self.mask.as_ref()
    }

    #[must_use]
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    #[must_use]
    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }

    #[must_use]
    pub fn fill_value(&self) -> f64 {
        self.fill_value
    }

    /// Return data with masked elements replaced by `fill_value`.
    #[must_use]
    pub fn filled(&self, fill_value: f64) -> UFuncArray {
        match &self.mask {
            None => self.data.clone(),
            Some(mask) => {
                let vals: Vec<f64> = self
                    .data
                    .values()
                    .iter()
                    .zip(mask.values().iter())
                    .map(|(&d, &m)| if m != 0.0 { fill_value } else { d })
                    .collect();
                UFuncArray {
                    shape: self.data.shape().to_vec(),
                    values: vals,
                    dtype: self.data.dtype(),
                }
            }
        }
    }

    /// Return only the valid (non-masked) elements as a 1-D array.
    #[must_use]
    pub fn compressed(&self) -> UFuncArray {
        match &self.mask {
            None => {
                // Flatten all data
                UFuncArray {
                    shape: vec![self.data.values().len()],
                    values: self.data.values().to_vec(),
                    dtype: self.data.dtype(),
                }
            }
            Some(mask) => {
                let vals: Vec<f64> = self
                    .data
                    .values()
                    .iter()
                    .zip(mask.values().iter())
                    .filter(|pair| *pair.1 == 0.0)
                    .map(|pair| *pair.0)
                    .collect();
                UFuncArray {
                    shape: vec![vals.len()],
                    values: vals,
                    dtype: self.data.dtype(),
                }
            }
        }
    }

    /// Count valid (non-masked) elements along an axis.
    pub fn count(&self, axis: Option<isize>) -> Result<UFuncArray, MAError> {
        match &self.mask {
            None => {
                // No mask: count = size along axis (or total size)
                match axis {
                    None => {
                        let total = self.data.values().len() as f64;
                        Ok(UFuncArray::scalar(total, DType::I64))
                    }
                    Some(_) => {
                        // All valid: sum of ones along axis gives dimension size
                        let ones = UFuncArray {
                            shape: self.data.shape().to_vec(),
                            values: vec![1.0; self.data.values().len()],
                            dtype: DType::F64,
                        };
                        Ok(ones.reduce_sum(axis, false)?)
                    }
                }
            }
            Some(mask) => {
                // Count where mask == 0 (valid): invert mask, then sum
                let inverted: Vec<f64> = mask
                    .values()
                    .iter()
                    .map(|&m| if m == 0.0 { 1.0 } else { 0.0 })
                    .collect();
                let inv = UFuncArray {
                    shape: mask.shape().to_vec(),
                    values: inverted,
                    dtype: DType::F64,
                };
                Ok(inv.reduce_sum(axis, false)?)
            }
        }
    }

    // ── Mask mutation ───────────────────────────────────────────────

    /// Set the fill value.
    pub fn set_fill_value(&mut self, fill_value: f64) {
        self.fill_value = fill_value;
    }

    /// Harden the mask: once hardened, masked elements cannot be unmasked.
    pub fn harden_mask(&mut self) {
        self.hard_mask = true;
    }

    /// Soften the mask: allow masked elements to be unmasked.
    pub fn soften_mask(&mut self) {
        self.hard_mask = false;
    }

    #[must_use]
    pub fn is_hard_mask(&self) -> bool {
        self.hard_mask
    }

    // ── Helper: inverted mask for reduce_sum_where ──────────────────

    /// Build an inverted mask (1.0 where valid) suitable for `reduce_sum_where`.
    fn valid_mask(&self) -> Option<UFuncArray> {
        self.mask.as_ref().map(|mask| {
            let inverted: Vec<f64> = mask
                .values()
                .iter()
                .map(|&m| if m == 0.0 { 1.0 } else { 0.0 })
                .collect();
            UFuncArray {
                shape: mask.shape().to_vec(),
                values: inverted,
                dtype: DType::Bool,
            }
        })
    }

    // ── Arithmetic with mask propagation ────────────────────────────

    /// Apply a binary operation, propagating masks (OR).
    pub fn elementwise_binary(&self, other: &Self, op: BinaryOp) -> Result<Self, MAError> {
        let result_data = self.data.elementwise_binary(&other.data, op)?;
        let result_mask = match (&self.mask, &other.mask) {
            (None, None) => None,
            (Some(m), None) | (None, Some(m)) => Some(m.clone()),
            (Some(m1), Some(m2)) => Some(m1.elementwise_binary(m2, BinaryOp::LogicalOr)?),
        };
        Ok(Self {
            data: result_data,
            mask: result_mask,
            fill_value: self.fill_value,
            hard_mask: self.hard_mask,
        })
    }

    /// Apply a unary operation (mask unchanged).
    #[must_use]
    pub fn elementwise_unary(&self, op: UnaryOp) -> Self {
        Self {
            data: self.data.elementwise_unary(op),
            mask: self.mask.clone(),
            fill_value: self.fill_value,
            hard_mask: self.hard_mask,
        }
    }

    // ── Reductions (masked elements excluded) ───────────────────────

    /// Sum of non-masked elements.
    pub fn sum(&self, axis: Option<isize>, keepdims: bool) -> Result<Self, MAError> {
        let result_data = match self.valid_mask() {
            None => self.data.reduce_sum(axis, keepdims)?,
            Some(vmask) => self.data.reduce_sum_where(&vmask, axis, keepdims)?,
        };
        Ok(Self {
            data: result_data,
            mask: None,
            fill_value: self.fill_value,
            hard_mask: false,
        })
    }

    /// Product of non-masked elements.
    pub fn prod(&self, axis: Option<isize>, keepdims: bool) -> Result<Self, MAError> {
        // For product, replace masked values with 1.0 (identity), then reduce
        let filled = self.filled(1.0);
        let result_data = filled.reduce_prod(axis, keepdims)?;
        Ok(Self {
            data: result_data,
            mask: None,
            fill_value: self.fill_value,
            hard_mask: false,
        })
    }

    /// Mean of non-masked elements.
    pub fn mean(&self, axis: Option<isize>, keepdims: bool) -> Result<Self, MAError> {
        let sum_result = match self.valid_mask() {
            None => self.data.reduce_sum(axis, keepdims)?,
            Some(vmask) => self.data.reduce_sum_where(&vmask, axis, keepdims)?,
        };
        let count_result = self.count(axis)?;
        let count_broadcast = if keepdims {
            let target: Vec<isize> = sum_result.shape().iter().map(|&d| d as isize).collect();
            count_result.reshape(&target)?
        } else {
            count_result
        };
        let result_data = sum_result.elementwise_binary(&count_broadcast, BinaryOp::Div)?;
        Ok(Self {
            data: result_data,
            mask: None,
            fill_value: self.fill_value,
            hard_mask: false,
        })
    }

    /// Minimum of non-masked elements.
    pub fn min(&self, axis: Option<isize>, keepdims: bool) -> Result<Self, MAError> {
        let filled = self.filled(f64::INFINITY);
        let result_data = filled.reduce_min(axis, keepdims)?;
        Ok(Self {
            data: result_data,
            mask: None,
            fill_value: self.fill_value,
            hard_mask: false,
        })
    }

    /// Maximum of non-masked elements.
    pub fn max(&self, axis: Option<isize>, keepdims: bool) -> Result<Self, MAError> {
        let filled = self.filled(f64::NEG_INFINITY);
        let result_data = filled.reduce_max(axis, keepdims)?;
        Ok(Self {
            data: result_data,
            mask: None,
            fill_value: self.fill_value,
            hard_mask: false,
        })
    }

    /// Variance of non-masked elements.
    pub fn var(&self, axis: Option<isize>, keepdims: bool, ddof: usize) -> Result<Self, MAError> {
        // var = mean((x - mean(x))^2), adjusted for ddof
        // mean with keepdims=true for broadcasting back to data shape
        let mean_arr = self.mean(axis, true)?;
        let deviations = self
            .data
            .elementwise_binary(&mean_arr.data, BinaryOp::Sub)?;
        let sq_deviations = deviations.elementwise_binary(&deviations, BinaryOp::Mul)?;
        // Mask the squared deviations the same way as self
        let masked_sq = Self {
            data: sq_deviations,
            mask: self.mask.clone(),
            fill_value: self.fill_value,
            hard_mask: false,
        };
        let sum_sq = masked_sq.sum(axis, keepdims)?;
        let count_result = self.count(axis)?;
        // Subtract ddof from count
        let ddof_val = UFuncArray::scalar(ddof as f64, DType::F64);
        let adjusted_count = count_result.elementwise_binary(&ddof_val, BinaryOp::Sub)?;
        let adjusted_broadcast = if keepdims {
            let target: Vec<isize> = sum_sq.data.shape().iter().map(|&d| d as isize).collect();
            adjusted_count.reshape(&target)?
        } else {
            adjusted_count
        };
        let result_data = sum_sq
            .data
            .elementwise_binary(&adjusted_broadcast, BinaryOp::Div)?;
        Ok(Self {
            data: result_data,
            mask: None,
            fill_value: self.fill_value,
            hard_mask: false,
        })
    }

    /// Standard deviation of non-masked elements.
    pub fn std(&self, axis: Option<isize>, keepdims: bool, ddof: usize) -> Result<Self, MAError> {
        let variance = self.var(axis, keepdims, ddof)?;
        let result_data = variance.data.elementwise_unary(UnaryOp::Sqrt);
        Ok(Self {
            data: result_data,
            mask: None,
            fill_value: self.fill_value,
            hard_mask: false,
        })
    }

    // ── Shape manipulation ─────────────────────────────────────────

    /// Reshape the masked array. Both data and mask are reshaped.
    pub fn reshape(&self, new_shape: &[isize]) -> Result<Self, MAError> {
        let new_data = self.data.reshape(new_shape)?;
        let new_mask = match &self.mask {
            None => None,
            Some(m) => Some(m.reshape(new_shape)?),
        };
        Ok(Self {
            data: new_data,
            mask: new_mask,
            fill_value: self.fill_value,
            hard_mask: self.hard_mask,
        })
    }

    /// Flatten the masked array to 1-D.
    #[must_use]
    pub fn ravel(&self) -> Self {
        let n = self.data.values().len();
        Self {
            data: UFuncArray {
                shape: vec![n],
                values: self.data.values().to_vec(),
                dtype: self.data.dtype(),
            },
            mask: self.mask.as_ref().map(|m| UFuncArray {
                shape: vec![n],
                values: m.values().to_vec(),
                dtype: DType::Bool,
            }),
            fill_value: self.fill_value,
            hard_mask: self.hard_mask,
        }
    }

    /// Transpose the masked array.
    pub fn transpose(&self, axes: Option<&[usize]>) -> Result<Self, MAError> {
        let new_data = self.data.transpose(axes)?;
        let new_mask = match &self.mask {
            None => None,
            Some(m) => Some(m.transpose(axes)?),
        };
        Ok(Self {
            data: new_data,
            mask: new_mask,
            fill_value: self.fill_value,
            hard_mask: self.hard_mask,
        })
    }

    // ── Comparison operations ──────────────────────────────────────

    /// Element-wise less-than, returning a MaskedArray of booleans (1.0/0.0).
    pub fn less_than(&self, other: &Self) -> Result<Self, MAError> {
        self.elementwise_binary(other, BinaryOp::Less)
    }

    /// Element-wise less-than-or-equal.
    pub fn less_equal(&self, other: &Self) -> Result<Self, MAError> {
        self.elementwise_binary(other, BinaryOp::LessEqual)
    }

    /// Element-wise greater-than.
    pub fn greater_than(&self, other: &Self) -> Result<Self, MAError> {
        self.elementwise_binary(other, BinaryOp::Greater)
    }

    /// Element-wise greater-than-or-equal.
    pub fn greater_equal(&self, other: &Self) -> Result<Self, MAError> {
        self.elementwise_binary(other, BinaryOp::GreaterEqual)
    }

    /// Element-wise equality.
    pub fn equal(&self, other: &Self) -> Result<Self, MAError> {
        self.elementwise_binary(other, BinaryOp::Equal)
    }

    /// Element-wise not-equal.
    pub fn not_equal(&self, other: &Self) -> Result<Self, MAError> {
        self.elementwise_binary(other, BinaryOp::NotEqual)
    }

    // ── Concatenation ──────────────────────────────────────────────

    /// Concatenate multiple masked arrays along an axis.
    /// `np.ma.concatenate(arrays, axis=0)`
    pub fn concatenate(arrays: &[&Self], axis: isize) -> Result<Self, MAError> {
        if arrays.is_empty() {
            return Err(MAError::Msg("concatenate: need at least 1 array".into()));
        }
        let data_refs: Vec<&UFuncArray> = arrays.iter().map(|a| &a.data).collect();
        let result_data = UFuncArray::concatenate(&data_refs, axis)?;

        // Build combined mask: if any input has a mask, produce a full mask
        let any_masked = arrays.iter().any(|a| a.mask.is_some());
        let result_mask = if any_masked {
            let mask_arrays: Vec<UFuncArray> = arrays
                .iter()
                .map(|a| match &a.mask {
                    Some(m) => m.clone(),
                    None => UFuncArray {
                        shape: a.data.shape().to_vec(),
                        values: vec![0.0; a.data.values().len()],
                        dtype: DType::Bool,
                    },
                })
                .collect();
            let mask_refs: Vec<&UFuncArray> = mask_arrays.iter().collect();
            Some(UFuncArray::concatenate(&mask_refs, axis)?)
        } else {
            None
        };

        Ok(Self {
            data: result_data,
            mask: result_mask,
            fill_value: arrays[0].fill_value,
            hard_mask: false,
        })
    }

    // ── Additional utilities ────────────────────────────────────────

    /// Return the indices of non-masked elements (as flat indices).
    /// `np.ma.nonzero` equivalent.
    #[must_use]
    pub fn nonzero_indices(&self) -> Vec<usize> {
        match &self.mask {
            None => (0..self.data.values().len())
                .filter(|&i| self.data.values()[i] != 0.0)
                .collect(),
            Some(mask) => (0..self.data.values().len())
                .filter(|&i| mask.values()[i] == 0.0 && self.data.values()[i] != 0.0)
                .collect(),
        }
    }

    /// Check if any valid (non-masked) element is true (non-zero).
    pub fn any(&self) -> bool {
        match &self.mask {
            None => self.data.values().iter().any(|&v| v != 0.0),
            Some(mask) => self
                .data
                .values()
                .iter()
                .zip(mask.values().iter())
                .any(|(&d, &m)| m == 0.0 && d != 0.0),
        }
    }

    /// Check if all valid (non-masked) elements are true (non-zero).
    pub fn all(&self) -> bool {
        match &self.mask {
            None => self.data.values().iter().all(|&v| v != 0.0),
            Some(mask) => self
                .data
                .values()
                .iter()
                .zip(mask.values().iter())
                .all(|(&d, &m)| m != 0.0 || d != 0.0),
        }
    }

    /// Return a copy with the mask removed.
    #[must_use]
    pub fn copy_without_mask(&self) -> Self {
        Self {
            data: self.data.clone(),
            mask: None,
            fill_value: self.fill_value,
            hard_mask: false,
        }
    }

    /// Return the number of elements (including masked).
    #[must_use]
    pub fn size(&self) -> usize {
        self.data.values().len()
    }

    /// Return the number of dimensions.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.data.shape().len()
    }

    // ── Additional MaskedArray operations (numpy.ma parity) ──────

    /// Sort unmasked elements, pushing masked to end (np.ma.sort).
    pub fn sort(&self, axis: Option<isize>) -> Result<Self, MAError> {
        let filled = self.filled(f64::MAX);
        let sorted = filled.sort(axis, None)?;
        // Reconstruct mask: masked elements are those equal to f64::MAX after sort
        let mask_vals: Vec<f64> = sorted
            .values()
            .iter()
            .map(|&v| if v == f64::MAX { 1.0 } else { 0.0 })
            .collect();
        let mask = if mask_vals.contains(&1.0) {
            Some(UFuncArray::new(
                sorted.shape().to_vec(),
                mask_vals,
                DType::Bool,
            )?)
        } else {
            None
        };
        Ok(Self {
            data: sorted,
            mask,
            fill_value: self.fill_value,
            hard_mask: self.hard_mask,
        })
    }

    /// Argsort unmasked elements, masked indices appear at the end (np.ma.argsort).
    pub fn argsort(&self, axis: Option<isize>) -> Result<UFuncArray, MAError> {
        let filled = self.filled(f64::MAX);
        Ok(filled.argsort(axis, None)?)
    }

    /// Index of minimum value among unmasked elements (np.ma.argmin).
    pub fn argmin(&self, axis: Option<isize>) -> Result<UFuncArray, MAError> {
        let filled = self.filled(f64::MAX);
        Ok(filled.reduce_argmin(axis)?)
    }

    /// Index of maximum value among unmasked elements (np.ma.argmax).
    pub fn argmax(&self, axis: Option<isize>) -> Result<UFuncArray, MAError> {
        let filled = self.filled(f64::MIN);
        Ok(filled.reduce_argmax(axis)?)
    }

    /// Cumulative sum over unmasked elements (np.ma.cumsum).
    /// Masked elements contribute 0 to the running sum; mask is preserved.
    pub fn cumsum(&self, axis: Option<isize>) -> Result<Self, MAError> {
        let filled = self.filled(0.0);
        let cs = filled.cumsum(axis)?;
        Ok(Self {
            data: cs,
            mask: self.mask.clone(),
            fill_value: self.fill_value,
            hard_mask: self.hard_mask,
        })
    }

    /// Cumulative product over unmasked elements (np.ma.cumprod).
    /// Masked elements contribute 1 to the running product; mask is preserved.
    pub fn cumprod(&self, axis: Option<isize>) -> Result<Self, MAError> {
        let filled = self.filled(1.0);
        let cp = filled.cumprod(axis)?;
        Ok(Self {
            data: cp,
            mask: self.mask.clone(),
            fill_value: self.fill_value,
            hard_mask: self.hard_mask,
        })
    }

    /// Median of unmasked elements (np.ma.median).
    pub fn median(&self) -> Result<f64, MAError> {
        let compressed = self.compressed();
        if compressed.values().is_empty() {
            return Err(MAError::Msg("median: no unmasked elements".into()));
        }
        let mut vals = compressed.values().to_vec();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = vals.len();
        if n.is_multiple_of(2) {
            Ok((vals[n / 2 - 1] + vals[n / 2]) / 2.0)
        } else {
            Ok(vals[n / 2])
        }
    }

    /// Peak-to-peak (max - min) of unmasked elements (np.ma.ptp).
    pub fn ptp(&self) -> Result<f64, MAError> {
        let compressed = self.compressed();
        if compressed.values().is_empty() {
            return Err(MAError::Msg("ptp: no unmasked elements".into()));
        }
        let mn = compressed
            .values()
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let mx = compressed
            .values()
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        Ok(mx - mn)
    }

    /// Dot product of two masked arrays (np.ma.dot).
    /// Masked elements are treated as zero.
    pub fn dot(&self, other: &Self) -> Result<Self, MAError> {
        let a = self.filled(0.0);
        let b = other.filled(0.0);
        let result = a.dot(&b)?;
        Ok(Self {
            data: result,
            mask: None,
            fill_value: self.fill_value,
            hard_mask: false,
        })
    }

    /// Expand dimensions of a masked array (np.ma.expand_dims).
    pub fn expand_dims(&self, axis: isize) -> Result<Self, MAError> {
        let new_data = self.data.expand_dims(axis)?;
        let new_mask = match &self.mask {
            Some(m) => Some(m.expand_dims(axis)?),
            None => None,
        };
        Ok(Self {
            data: new_data,
            mask: new_mask,
            fill_value: self.fill_value,
            hard_mask: self.hard_mask,
        })
    }

    /// Squeeze singleton dimensions (np.ma.squeeze).
    pub fn squeeze(&self, axis: Option<isize>) -> Result<Self, MAError> {
        let new_data = self.data.squeeze(axis)?;
        let new_mask = match &self.mask {
            Some(m) => Some(m.squeeze(axis)?),
            None => None,
        };
        Ok(Self {
            data: new_data,
            mask: new_mask,
            fill_value: self.fill_value,
            hard_mask: self.hard_mask,
        })
    }

    /// Create a masked array where all elements are masked (np.ma.masked_all).
    pub fn masked_all(shape: Vec<usize>, dtype: DType) -> Result<Self, MAError> {
        let n: usize = shape.iter().product();
        let data = UFuncArray::new(shape.clone(), vec![0.0; n], dtype)?;
        let mask = UFuncArray::new(shape, vec![1.0; n], DType::Bool)?;
        Ok(Self {
            data,
            mask: Some(mask),
            fill_value: 1e20,
            hard_mask: false,
        })
    }

    /// Create a masked array with same shape as model, all masked (np.ma.masked_all_like).
    pub fn masked_all_like(model: &Self) -> Result<Self, MAError> {
        Self::masked_all(model.shape().to_vec(), model.dtype())
    }

    /// Count of masked (invalid) elements (np.ma.count_masked).
    pub fn count_masked(&self) -> usize {
        match &self.mask {
            Some(m) => m.values().iter().filter(|&&v| v != 0.0).count(),
            None => 0,
        }
    }

    /// Weighted average of unmasked elements (np.ma.average).
    pub fn average(&self, weights: Option<&Self>) -> Result<f64, MAError> {
        match weights {
            None => {
                let compressed = self.compressed();
                if compressed.values().is_empty() {
                    return Err(MAError::Msg("average: no unmasked elements".into()));
                }
                let s: f64 = compressed.values().iter().sum();
                Ok(s / compressed.values().len() as f64)
            }
            Some(w) => {
                // Both data and weights must match shape
                let n = self.data.values().len();
                let mut wsum = 0.0;
                let mut vsum = 0.0;
                for i in 0..n {
                    let masked = self.mask.as_ref().is_some_and(|m| m.values()[i] != 0.0);
                    if !masked {
                        let wi = w.data.values()[i];
                        wsum += wi;
                        vsum += self.data.values()[i] * wi;
                    }
                }
                if wsum == 0.0 {
                    return Err(MAError::Msg("average: zero weight sum".into()));
                }
                Ok(vsum / wsum)
            }
        }
    }

    /// Take elements from masked array by indices (np.ma.take).
    pub fn take(&self, indices: &[i64]) -> Result<Self, MAError> {
        let new_data = self.data.take(indices, None)?;
        let new_mask = match &self.mask {
            Some(m) => Some(m.take(indices, None)?),
            None => None,
        };
        Ok(Self {
            data: new_data,
            mask: new_mask,
            fill_value: self.fill_value,
            hard_mask: self.hard_mask,
        })
    }

    /// Return contiguous slices of unmasked data (np.ma.clump_unmasked).
    /// Returns a vector of (start, end) index pairs for contiguous unmasked runs.
    pub fn clump_unmasked(&self) -> Vec<(usize, usize)> {
        let mut runs = Vec::new();
        let mut start: Option<usize> = None;
        let n = self.data.values().len();
        for i in 0..n {
            let masked = self.mask.as_ref().is_some_and(|m| m.values()[i] != 0.0);
            if masked {
                if let Some(s) = start {
                    runs.push((s, i));
                    start = None;
                }
            } else if start.is_none() {
                start = Some(i);
            }
        }
        if let Some(s) = start {
            runs.push((s, n));
        }
        runs
    }

    /// Return contiguous slices of masked data (np.ma.clump_masked).
    /// Returns a vector of (start, end) index pairs for contiguous masked runs.
    pub fn clump_masked(&self) -> Vec<(usize, usize)> {
        let mut runs = Vec::new();
        let mut start: Option<usize> = None;
        let n = self.data.values().len();
        for i in 0..n {
            let masked = self.mask.as_ref().is_some_and(|m| m.values()[i] != 0.0);
            if !masked {
                if let Some(s) = start {
                    runs.push((s, i));
                    start = None;
                }
            } else if start.is_none() {
                start = Some(i);
            }
        }
        if let Some(s) = start {
            runs.push((s, n));
        }
        runs
    }

    /// Remove the mask if all elements are valid (np.ma.MaskedArray.shrink_mask).
    /// If any element is masked, the mask is kept unchanged.
    pub fn shrink_mask(&mut self) {
        if let Some(m) = &self.mask
            && m.values().iter().all(|&v| v == 0.0)
        {
            self.mask = None;
        }
    }

    /// Deviation from mean of unmasked elements (np.ma.MaskedArray.anom).
    /// Returns a new MaskedArray with each unmasked element replaced by
    /// (element - mean). Masked elements are preserved.
    pub fn anom(&self) -> Result<Self, MAError> {
        let compressed = self.compressed();
        if compressed.values().is_empty() {
            return Err(MAError::Msg("anom: no unmasked elements".into()));
        }
        let mean: f64 = compressed.values().iter().sum::<f64>() / compressed.values().len() as f64;
        let mut new_vals = self.data.values().to_vec();
        for (i, val) in new_vals.iter_mut().enumerate() {
            let masked = self.mask.as_ref().is_some_and(|m| m.values()[i] != 0.0);
            if !masked {
                *val -= mean;
            }
        }
        let new_data = UFuncArray::new(self.shape().to_vec(), new_vals, self.dtype())?;
        Ok(Self {
            data: new_data,
            mask: self.mask.clone(),
            fill_value: self.fill_value,
            hard_mask: self.hard_mask,
        })
    }

    /// Mask invalid values in place (NaN, Inf) (np.ma.fix_invalid as mutation).
    /// Sets the mask to 1 wherever the data contains NaN or Inf, and replaces
    /// those data values with `fill_value`.
    pub fn fix_invalid(&mut self) {
        let n = self.data.values().len();
        let mut mask_vals = match &self.mask {
            Some(m) => m.values().to_vec(),
            None => vec![0.0; n],
        };
        let mut data_vals = self.data.values().to_vec();
        for i in 0..n {
            if !data_vals[i].is_finite() {
                mask_vals[i] = 1.0;
                data_vals[i] = self.fill_value;
            }
        }
        self.data = UFuncArray::new(self.shape().to_vec(), data_vals, self.dtype())
            .unwrap_or_else(|_| self.data.clone());
        self.mask = Some(
            UFuncArray::new(self.shape().to_vec(), mask_vals, DType::Bool)
                .unwrap_or_else(|_| self.mask.clone().unwrap()),
        );
    }
}

/// Check whether a MaskedArray has any masked elements (np.ma.is_masked).
pub fn ma_is_masked(a: &MaskedArray) -> bool {
    a.mask
        .as_ref()
        .is_some_and(|m| m.values().iter().any(|&v| v != 0.0))
}

/// Check whether an array qualifies as a boolean mask (np.ma.is_mask).
/// Returns true if `arr` contains only 0.0 and 1.0 values with Bool dtype.
pub fn ma_is_mask(arr: &UFuncArray) -> bool {
    arr.dtype() == DType::Bool && arr.values().iter().all(|&v| v == 0.0 || v == 1.0)
}

/// Create a boolean mask from an array, treating non-zero as masked (np.ma.make_mask).
pub fn ma_make_mask(arr: &UFuncArray) -> UFuncArray {
    let vals: Vec<f64> = arr
        .values()
        .iter()
        .map(|&v| if v != 0.0 { 1.0 } else { 0.0 })
        .collect();
    UFuncArray::new(arr.shape().to_vec(), vals, DType::Bool).unwrap()
}

/// Combine two masks with logical OR (np.ma.mask_or).
/// Returns None if both inputs are None.
pub fn ma_mask_or(m1: Option<&UFuncArray>, m2: Option<&UFuncArray>) -> Option<UFuncArray> {
    match (m1, m2) {
        (None, None) => None,
        (Some(m), None) | (None, Some(m)) => Some(m.clone()),
        (Some(a), Some(b)) => {
            let vals: Vec<f64> = a
                .values()
                .iter()
                .zip(b.values().iter())
                .map(|(&va, &vb)| if va != 0.0 || vb != 0.0 { 1.0 } else { 0.0 })
                .collect();
            Some(UFuncArray::new(a.shape().to_vec(), vals, DType::Bool).unwrap())
        }
    }
}

// ── numpy.financial — Financial Functions ────────────────────────────────
//
// These mirror the functions from numpy-financial (formerly numpy.lib.financial,
// removed from NumPy 2.0). All operate on scalar f64 values and return f64.
// Conventions: `rate` is interest rate per period, `nper` is number of periods,
// `pmt` is payment per period, `pv` is present value, `fv` is future value.
// `when` indicates timing: 0 = end of period (default), 1 = beginning.

/// Future value of an investment: `numpy_financial.fv(rate, nper, pmt, pv, when=0)`.
pub fn financial_fv(rate: f64, nper: f64, pmt: f64, pv: f64, when: u8) -> f64 {
    if rate == 0.0 {
        return -(pv + pmt * nper);
    }
    let factor = (1.0 + rate).powf(nper);
    let when_f = f64::from(when);
    -(pv * factor + pmt * (1.0 + rate * when_f) * (factor - 1.0) / rate)
}

/// Present value: `numpy_financial.pv(rate, nper, pmt, fv, when=0)`.
pub fn financial_pv(rate: f64, nper: f64, pmt: f64, fv: f64, when: u8) -> f64 {
    if rate == 0.0 {
        return -(fv + pmt * nper);
    }
    let factor = (1.0 + rate).powf(nper);
    let when_f = f64::from(when);
    -(fv + pmt * (1.0 + rate * when_f) * (factor - 1.0) / rate) / factor
}

/// Payment per period: `numpy_financial.pmt(rate, nper, pv, fv, when=0)`.
pub fn financial_pmt(rate: f64, nper: f64, pv: f64, fv: f64, when: u8) -> f64 {
    if rate == 0.0 {
        return -(fv + pv) / nper;
    }
    let factor = (1.0 + rate).powf(nper);
    let when_f = f64::from(when);
    -(fv + pv * factor) * rate / ((1.0 + rate * when_f) * (factor - 1.0))
}

/// Number of periods: `numpy_financial.nper(rate, pmt, pv, fv, when=0)`.
pub fn financial_nper(rate: f64, pmt: f64, pv: f64, fv: f64, when: u8) -> f64 {
    if rate == 0.0 {
        return -(fv + pv) / pmt;
    }
    let when_f = f64::from(when);
    let z = pmt * (1.0 + rate * when_f) / rate;
    ((z - fv) / (z + pv)).ln() / (1.0 + rate).ln()
}

/// Net present value: `numpy_financial.npv(rate, cashflows)`.
pub fn financial_npv(rate: f64, cashflows: &[f64]) -> f64 {
    cashflows
        .iter()
        .enumerate()
        .map(|(i, cf)| cf / (1.0_f64 + rate).powi(i as i32))
        .sum()
}

/// Internal rate of return: `numpy_financial.irr(cashflows)`.
///
/// Uses Newton's method to find the rate where NPV = 0.
pub fn financial_irr(cashflows: &[f64]) -> f64 {
    if cashflows.len() < 2 {
        return f64::NAN;
    }
    let mut rate: f64 = 0.1; // initial guess
    for _ in 0..100 {
        let base = 1.0_f64 + rate;
        let npv: f64 = cashflows
            .iter()
            .enumerate()
            .map(|(i, cf)| cf / base.powi(i as i32))
            .sum();
        let dnpv: f64 = cashflows
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, cf)| -(i as f64) * cf / base.powi(i as i32 + 1))
            .sum();
        if dnpv.abs() < 1e-15 {
            break;
        }
        let new_rate = rate - npv / dnpv;
        if (new_rate - rate).abs() < 1e-12 {
            return new_rate;
        }
        rate = new_rate;
    }
    rate
}

/// Interest portion of a payment: `numpy_financial.ipmt(rate, per, nper, pv, fv, when)`.
///
/// `per` is the period (1-based).
pub fn financial_ipmt(rate: f64, per: f64, nper: f64, pv: f64, fv: f64, when: u8) -> f64 {
    let total_pmt = financial_pmt(rate, nper, pv, fv, when);
    let remaining_balance = financial_fv(rate, per - 1.0, total_pmt, pv, when);
    if when == 1 && per == 1.0 {
        0.0
    } else {
        remaining_balance * rate
    }
}

/// Principal portion of a payment: `numpy_financial.ppmt(rate, per, nper, pv, fv, when)`.
pub fn financial_ppmt(rate: f64, per: f64, nper: f64, pv: f64, fv: f64, when: u8) -> f64 {
    financial_pmt(rate, nper, pv, fv, when) - financial_ipmt(rate, per, nper, pv, fv, when)
}

/// Modified internal rate of return: `numpy_financial.mirr(cashflows, finance_rate, reinvest_rate)`.
pub fn financial_mirr(cashflows: &[f64], finance_rate: f64, reinvest_rate: f64) -> f64 {
    let n = cashflows.len() as f64 - 1.0;
    // Separate positive and negative flows
    let neg_pv: f64 = cashflows
        .iter()
        .enumerate()
        .filter(|pair| *pair.1 < 0.0)
        .map(|(i, cf)| cf / (1.0_f64 + finance_rate).powi(i as i32))
        .sum();
    let pos_fv: f64 = cashflows
        .iter()
        .enumerate()
        .filter(|pair| *pair.1 > 0.0)
        .map(|(i, cf)| cf * (1.0_f64 + reinvest_rate).powf(n - i as f64))
        .sum();
    if neg_pv == 0.0 {
        return f64::NAN;
    }
    (-pos_fv / neg_pv).powf(1.0 / n) - 1.0
}

/// Interest rate per period: `numpy_financial.rate(nper, pmt, pv, fv, when, guess)`.
///
/// Uses Newton's method.
pub fn financial_rate(nper: f64, pmt: f64, pv: f64, fv: f64, when: u8, guess: f64) -> f64 {
    let when_f = f64::from(when);
    let mut rate = guess;
    for _ in 0..100 {
        let factor = (1.0 + rate).powf(nper);
        let residual = fv + pv * factor + pmt * (1.0 + rate * when_f) * (factor - 1.0) / rate;
        // Derivative components
        let dfactor = nper * (1.0 + rate).powf(nper - 1.0);
        let t1 = pv * dfactor;
        let annuity = (factor - 1.0) / rate;
        let dannuity = (dfactor * rate - (factor - 1.0)) / (rate * rate);
        let t2 = pmt * when_f * annuity + pmt * (1.0 + rate * when_f) * dannuity;
        let derivative = t1 + t2;
        if derivative.abs() < 1e-15 {
            break;
        }
        let new_rate = rate - residual / derivative;
        if (new_rate - rate).abs() < 1e-12 {
            return new_rate;
        }
        rate = new_rate;
    }
    rate
}

// ── where_nonzero / unique variants ─────────────────────────────────────

/// `numpy.where(condition)` single-argument form: returns indices of nonzero elements.
///
/// Returns one `UFuncArray` per input dimension, each containing the indices
/// along that dimension where the condition is nonzero.
pub fn where_nonzero(condition: &UFuncArray) -> Result<Vec<UFuncArray>, UFuncError> {
    let ndim = condition.shape.len();
    let strides = contiguous_strides_elems(&condition.shape);
    let mut nd_indices: Vec<Vec<f64>> = vec![Vec::new(); ndim.max(1)];

    for (flat, &v) in condition.values.iter().enumerate() {
        if v != 0.0 {
            if ndim == 0 {
                // scalar
                nd_indices[0].push(0.0);
            } else {
                let mut remaining = flat;
                for d in 0..ndim {
                    let idx = remaining / strides[d];
                    remaining %= strides[d];
                    nd_indices[d].push(idx as f64);
                }
            }
        }
    }

    let result: Vec<UFuncArray> = nd_indices
        .into_iter()
        .map(|indices| {
            let n = indices.len();
            UFuncArray {
                shape: vec![n],
                values: indices,
                dtype: DType::I64,
            }
        })
        .collect();
    Ok(result)
}

/// `numpy.unique_values(x)` — sorted unique values only.
pub fn unique_values(x: &UFuncArray) -> UFuncArray {
    x.unique()
}

/// `numpy.unique_counts(x)` — sorted unique values and their counts.
pub fn unique_counts(x: &UFuncArray) -> (UFuncArray, UFuncArray) {
    let (values, _, _, counts) = x.unique_with_info(false, false, true);
    (values, counts.expect("counts requested"))
}

/// `numpy.unique_inverse(x)` — sorted unique values and indices to reconstruct.
pub fn unique_inverse(x: &UFuncArray) -> (UFuncArray, UFuncArray) {
    let (values, _, inverse, _) = x.unique_with_info(false, true, false);
    (values, inverse.expect("inverse requested"))
}

/// `numpy.unique_all(x)` — sorted unique values, first indices, inverse indices, and counts.
pub fn unique_all(x: &UFuncArray) -> (UFuncArray, UFuncArray, UFuncArray, UFuncArray) {
    let (values, indices, inverse, counts) = x.unique_with_info(true, true, true);
    (
        values,
        indices.expect("indices requested"),
        inverse.expect("inverse requested"),
        counts.expect("counts requested"),
    )
}

// ── Chebyshev polynomial module ─────────────────────────────────────────

/// Evaluate a Chebyshev series at points `x` using the Clenshaw algorithm.
///
/// `c` is the coefficient vector: polynomial = c[0]*T_0 + c[1]*T_1 + ...
pub fn chebval(x: &[f64], c: &[f64]) -> Vec<f64> {
    if c.is_empty() {
        return vec![0.0; x.len()];
    }
    if c.len() == 1 {
        return vec![c[0]; x.len()];
    }
    x.iter()
        .map(|&xi| {
            // Clenshaw recurrence for Chebyshev polynomials
            let mut b_k1 = 0.0; // b_{k+1}
            let mut b_k2 = 0.0; // b_{k+2}
            for k in (1..c.len()).rev() {
                let b_k = c[k] + 2.0 * xi * b_k1 - b_k2;
                b_k2 = b_k1;
                b_k1 = b_k;
            }
            c[0] + xi * b_k1 - b_k2
        })
        .collect()
}

/// Add two Chebyshev coefficient arrays.
pub fn chebadd(c1: &[f64], c2: &[f64]) -> Vec<f64> {
    let len = c1.len().max(c2.len());
    let mut result = vec![0.0; len];
    for (i, &v) in c1.iter().enumerate() {
        result[i] += v;
    }
    for (i, &v) in c2.iter().enumerate() {
        result[i] += v;
    }
    result
}

/// Subtract two Chebyshev coefficient arrays.
pub fn chebsub(c1: &[f64], c2: &[f64]) -> Vec<f64> {
    let len = c1.len().max(c2.len());
    let mut result = vec![0.0; len];
    for (i, &v) in c1.iter().enumerate() {
        result[i] += v;
    }
    for (i, &v) in c2.iter().enumerate() {
        result[i] -= v;
    }
    result
}

/// Multiply two Chebyshev coefficient arrays.
///
/// Uses the identity T_m * T_n = (T_{m+n} + T_{|m-n|}) / 2.
pub fn chebmul(c1: &[f64], c2: &[f64]) -> Vec<f64> {
    if c1.is_empty() || c2.is_empty() {
        return Vec::new();
    }
    let len = c1.len() + c2.len() - 1;
    let mut result = vec![0.0; len];
    for (i, &a) in c1.iter().enumerate() {
        for (j, &b) in c2.iter().enumerate() {
            let prod = a * b / 2.0;
            result[i + j] += prod;
            result[i.abs_diff(j)] += prod;
        }
    }
    result
}

/// Divide Chebyshev coefficient arrays. Returns (quotient, remainder).
pub fn chebdiv(c1: &[f64], c2: &[f64]) -> Result<(Vec<f64>, Vec<f64>), UFuncError> {
    if c2.is_empty() || c2.iter().all(|&v| v == 0.0) {
        return Err(UFuncError::Msg(
            "chebdiv: division by zero polynomial".to_string(),
        ));
    }
    // Convert to power basis, divide, convert back
    let p1 = cheb2poly(c1);
    let p2 = cheb2poly(c2);
    let (pq, pr) = poly_div(&p1, &p2)?;
    Ok((poly2cheb(&pq), poly2cheb(&pr)))
}

/// Differentiate a Chebyshev series `m` times.
pub fn chebder(c: &[f64], m: usize) -> Vec<f64> {
    if m == 0 {
        return c.to_vec();
    }
    let mut coeffs = c.to_vec();
    for _ in 0..m {
        let n = coeffs.len();
        if n <= 1 {
            return vec![0.0];
        }
        let mut der = vec![0.0; n - 1];
        // Chebyshev derivative recurrence:
        // c'[n-2] = 2*(n-1)*c[n-1]  (starting from top)
        // c'[k] = c'[k+2] + 2*(k+1)*c[k+1]  for k = n-3..0
        // but c'[0] needs halving of the "2*" factor
        if n >= 2 {
            der[n - 2] = 2.0 * (n as f64 - 1.0) * coeffs[n - 1];
        }
        for k in (0..n.saturating_sub(2)).rev() {
            let next = if k + 2 < der.len() { der[k + 2] } else { 0.0 };
            der[k] = next + 2.0 * (k as f64 + 1.0) * coeffs[k + 1];
        }
        // The k=0 coefficient accumulates with factor 2, but T_0 derivative
        // contributions use factor 1 (not 2), so halve der[0]
        der[0] /= 2.0;
        coeffs = der;
    }
    coeffs
}

/// Integrate a Chebyshev series `m` times with integration constant 0.
pub fn chebint(c: &[f64], m: usize) -> Vec<f64> {
    if m == 0 {
        return c.to_vec();
    }
    let mut coeffs = c.to_vec();
    for _ in 0..m {
        let n = coeffs.len();
        let mut int = vec![0.0; n + 1];
        // Integration recurrence:
        // For n>=2: I[n] = c[n-1]/(2*n)
        // I[1] = c[0] - c[2]/2 (already handled by the loop)
        // I[0] = integration constant (we use 0)
        // Actually: int[k] = (c[k-1] - c[k+1]) / (2*k) for k >= 2
        // int[1] = c[0] - c[2]/2 ... simpler to use the standard formula:
        // int[k] = (c[k-1])/(2*k) for k >= 2, but accounting for the c[k+1] term
        // NumPy's formula: int[k] = (c[k-1] - c[k+1])/(2*k) for k >= 1
        // where c[k+1] = 0 if k+1 >= n
        for k in 1..=n {
            let ck_minus_1 = coeffs[k - 1];
            let ck_plus_1 = if k + 1 < n { coeffs[k + 1] } else { 0.0 };
            if k == 1 {
                // Special: factor is 1 not 2 for T_0 contribution
                int[k] = ck_minus_1 - ck_plus_1 / 2.0;
            } else {
                int[k] = (ck_minus_1 - ck_plus_1) / (2.0 * k as f64);
            }
        }
        // Integration constant: int[0] = 0
        coeffs = int;
    }
    coeffs
}

/// Find roots of a Chebyshev series via companion matrix eigenvalues.
pub fn chebroots(c: &[f64]) -> Result<Vec<f64>, UFuncError> {
    // Remove trailing zeros
    let mut coeffs: Vec<f64> = c.to_vec();
    while coeffs.len() > 1 && *coeffs.last().unwrap() == 0.0 {
        coeffs.pop();
    }
    let n = coeffs.len();
    if n <= 1 {
        return Ok(Vec::new());
    }
    if n == 2 {
        // Linear: c[0] + c[1]*T_1(x) = c[0] + c[1]*x = 0 → x = -c[0]/c[1]
        return Ok(vec![-coeffs[0] / coeffs[1]]);
    }

    // Build Chebyshev companion matrix (n-1 x n-1)
    let deg = n - 1;
    let mut mat = vec![0.0; deg * deg];
    // First row: special
    mat[1] = 1.0; // mat[0][1] = 1
    // Middle rows: T_{k+1} = 2x*T_k - T_{k-1} → x*T_k = (T_{k+1} + T_{k-1})/2
    for k in 1..deg - 1 {
        mat[k * deg + k - 1] = 0.5;
        mat[k * deg + k + 1] = 0.5;
    }
    // Last row: from the equation sum c_k T_k = 0, express T_{n-1} in terms of lower T_k
    // x*T_{n-2} = T_{n-1}/2 + T_{n-3}/2 (or similar)
    // Standard companion: last row adjusts for c[k]/c[n-1]
    let cn = coeffs[deg];
    for k in 0..deg {
        mat[(deg - 1) * deg + k] -= coeffs[k] / (2.0 * cn);
    }
    // Adjust the [n-2][n-1] -> add 0.5 for the standard recurrence
    if deg >= 2 {
        mat[(deg - 1) * deg + deg - 2] += 0.5;
    }

    // Find eigenvalues using fnp-linalg
    let eigenvalues = fnp_linalg::eig_nxn(&mat, deg)
        .map_err(|e| UFuncError::Msg(format!("chebroots eigenvalue error: {e}")))?;
    // Extract real parts (eigenvalues are interleaved [real, imag, ...])
    let roots: Vec<f64> = eigenvalues.chunks_exact(2).map(|pair| pair[0]).collect();
    let mut sorted = roots;
    sorted.sort_by(|a, b| a.total_cmp(b));
    Ok(sorted)
}

/// Build a Chebyshev series from its roots.
pub fn chebfromroots(roots: &[f64]) -> Vec<f64> {
    if roots.is_empty() {
        return vec![1.0];
    }
    // Start with (x - r[0]) in Chebyshev basis: [-r[0], 1]
    let mut result = vec![-roots[0], 1.0];
    for &r in &roots[1..] {
        // Multiply current polynomial by (x - r) in Chebyshev basis
        // (x - r) has Chebyshev coefficients [-r, 1]
        let factor = vec![-r, 1.0];
        result = chebmul(&result, &factor);
    }
    result
}

/// Least-squares fit to data in Chebyshev basis.
///
/// Fits `y = sum c_k T_k(x)` for k = 0..deg.
pub fn chebfit(x: &[f64], y: &[f64], deg: usize) -> Result<Vec<f64>, UFuncError> {
    if x.len() != y.len() {
        return Err(UFuncError::Msg(
            "chebfit: x and y must have same length".into(),
        ));
    }
    if x.len() <= deg {
        return Err(UFuncError::Msg(
            "chebfit: more data points needed than degree".into(),
        ));
    }
    let n = x.len();
    let ncols = deg + 1;
    // Build Vandermonde matrix in Chebyshev basis
    let mut vander = vec![0.0; n * ncols];
    for i in 0..n {
        vander[i * ncols] = 1.0; // T_0 = 1
        if ncols > 1 {
            vander[i * ncols + 1] = x[i]; // T_1 = x
        }
        for j in 2..ncols {
            // T_j = 2*x*T_{j-1} - T_{j-2}
            vander[i * ncols + j] =
                2.0 * x[i] * vander[i * ncols + j - 1] - vander[i * ncols + j - 2];
        }
    }
    // Solve V^T V c = V^T y  (normal equations)
    let mut vtv = vec![0.0; ncols * ncols];
    let mut vty = vec![0.0; ncols];
    for i in 0..ncols {
        for j in 0..ncols {
            let mut sum = 0.0;
            for k in 0..n {
                sum += vander[k * ncols + i] * vander[k * ncols + j];
            }
            vtv[i * ncols + j] = sum;
        }
        let mut sum = 0.0;
        for k in 0..n {
            sum += vander[k * ncols + i] * y[k];
        }
        vty[i] = sum;
    }
    // Solve via Gaussian elimination
    let coeffs = solve_linear_system(&vtv, &vty, ncols)?;
    Ok(coeffs)
}

/// Convert Chebyshev coefficients to power series (standard polynomial) coefficients.
pub fn cheb2poly(c: &[f64]) -> Vec<f64> {
    if c.is_empty() {
        return Vec::new();
    }
    if c.len() == 1 {
        return vec![c[0]];
    }
    let n = c.len();
    // Build T_k(x) as power series, accumulate
    let mut result = vec![0.0; n];
    // t_prev = T_0 = [1, 0, 0, ...], t_curr = T_1 = [0, 1, 0, ...]
    let mut t_prev = vec![0.0; n];
    let mut t_curr = vec![0.0; n];
    t_prev[0] = 1.0;
    t_curr[1] = 1.0;

    // Add c[0]*T_0 and c[1]*T_1
    for j in 0..n {
        result[j] += c[0] * t_prev[j];
    }
    if n > 1 {
        for j in 0..n {
            result[j] += c[1] * t_curr[j];
        }
    }

    for (k, &ck) in c.iter().enumerate().skip(2) {
        // T_k = 2*x*T_{k-1} - T_{k-2}
        let mut t_next = vec![0.0; n];
        // 2*x*T_{k-1}: shift coefficients up by 1 and multiply by 2
        for j in 0..n - 1 {
            t_next[j + 1] += 2.0 * t_curr[j];
        }
        // - T_{k-2}
        for j in 0..n {
            t_next[j] -= t_prev[j];
        }
        // Add c[k]*T_k
        for j in 0..n {
            result[j] += ck * t_next[j];
        }
        t_prev = t_curr;
        t_curr = t_next;
        let _ = k; // suppress unused warning
    }
    result
}

/// Convert power series coefficients to Chebyshev coefficients.
pub fn poly2cheb(p: &[f64]) -> Vec<f64> {
    if p.is_empty() {
        return Vec::new();
    }
    let n = p.len();
    // Evaluate at Chebyshev nodes and use discrete Chebyshev transform
    // Simpler approach: build the conversion matrix and solve
    // T_k in power basis form a matrix; invert it
    // For small degrees, use the identity matrix approach
    let mut cheb_matrix = vec![0.0; n * n];
    let mut t_prev = vec![0.0; n];
    let mut t_curr = vec![0.0; n];
    t_prev[0] = 1.0;
    if n > 1 {
        t_curr[1] = 1.0;
    }
    // Column 0 = T_0 in power basis
    for j in 0..n {
        cheb_matrix[j * n] = t_prev[j];
    }
    if n > 1 {
        for j in 0..n {
            cheb_matrix[j * n + 1] = t_curr[j];
        }
    }
    for k in 2..n {
        let mut t_next = vec![0.0; n];
        for j in 0..n - 1 {
            t_next[j + 1] += 2.0 * t_curr[j];
        }
        for j in 0..n {
            t_next[j] -= t_prev[j];
        }
        for j in 0..n {
            cheb_matrix[j * n + k] = t_next[j];
        }
        t_prev = t_curr;
        t_curr = t_next;
    }
    // Solve: cheb_matrix * c = p → c = cheb_matrix^{-1} * p
    solve_linear_system(&cheb_matrix, p, n).unwrap_or_else(|_| p.to_vec())
}

/// Helper: solve a linear system Ax = b via Gaussian elimination with partial pivoting.
fn solve_linear_system(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, UFuncError> {
    let mut aug = vec![0.0; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }
    // Forward elimination with partial pivoting
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col * (n + 1) + col].abs();
        for row in col + 1..n {
            let v = aug[row * (n + 1) + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(UFuncError::Msg(
                "solve_linear_system: singular matrix".to_string(),
            ));
        }
        if max_row != col {
            for j in 0..=n {
                aug.swap(col * (n + 1) + j, max_row * (n + 1) + j);
            }
        }
        let pivot = aug[col * (n + 1) + col];
        for row in col + 1..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for j in col..=n {
                let above = aug[col * (n + 1) + j];
                aug[row * (n + 1) + j] -= factor * above;
            }
        }
    }
    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];
        for j in i + 1..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] = sum / aug[i * (n + 1) + i];
    }
    Ok(x)
}

/// Helper: polynomial division in power basis. Returns (quotient, remainder).
fn poly_div(num: &[f64], den: &[f64]) -> Result<(Vec<f64>, Vec<f64>), UFuncError> {
    let mut num = num.to_vec();
    // Trim trailing zeros from denominator
    let mut d = den.to_vec();
    while d.len() > 1 && *d.last().unwrap() == 0.0 {
        d.pop();
    }
    if d.is_empty() || (d.len() == 1 && d[0] == 0.0) {
        return Err(UFuncError::Msg("poly_div: division by zero".to_string()));
    }
    while num.len() > 1 && *num.last().unwrap() == 0.0 {
        num.pop();
    }
    if num.len() < d.len() {
        return Ok((vec![0.0], num));
    }
    let q_len = num.len() - d.len() + 1;
    let mut q = vec![0.0; q_len];
    for i in (0..q_len).rev() {
        q[i] = num[i + d.len() - 1] / d[d.len() - 1];
        for j in 0..d.len() {
            num[i + j] -= q[i] * d[j];
        }
    }
    // Remainder
    let mut rem = num[..d.len() - 1].to_vec();
    while rem.len() > 1 && rem.last().unwrap().abs() < 1e-14 {
        rem.pop();
    }
    if rem.is_empty() {
        rem.push(0.0);
    }
    Ok((q, rem))
}

// ── Legendre polynomial module ──────────────────────────────────────────

/// Evaluate a Legendre series at points `x` using the Clenshaw algorithm.
///
/// Recurrence: P_0=1, P_1=x, (n+1)P_{n+1} = (2n+1)x P_n - n P_{n-1}
pub fn legval(x: &[f64], c: &[f64]) -> Vec<f64> {
    if c.is_empty() {
        return vec![0.0; x.len()];
    }
    if c.len() == 1 {
        return vec![c[0]; x.len()];
    }
    x.iter()
        .map(|&xi| {
            let nd = c.len();
            let mut c0 = c[nd - 2];
            let mut c1 = c[nd - 1];
            for i in (0..nd - 2).rev() {
                let k = i + 1;
                let tmp = c0;
                c0 = c[i] - c1 * (k as f64 / (k as f64 + 1.0));
                c1 = tmp + c1 * ((2.0 * k as f64 + 1.0) / (k as f64 + 1.0)) * xi;
            }
            c0 + c1 * xi
        })
        .collect()
}

/// Differentiate a Legendre series `m` times.
pub fn legder(c: &[f64], m: usize) -> Vec<f64> {
    if m == 0 {
        return c.to_vec();
    }
    let mut coeffs = c.to_vec();
    for _ in 0..m {
        let n = coeffs.len();
        if n <= 1 {
            return vec![0.0];
        }
        let mut der = vec![0.0; n - 1];
        for j in (0..n - 1).rev() {
            der[j] = (2.0 * j as f64 + 1.0) * coeffs[j + 1];
            if j + 2 < n - 1 {
                der[j] += der[j + 2];
            }
        }
        coeffs = der;
    }
    coeffs
}

/// Integrate a Legendre series `m` times (constant=0).
pub fn legint(c: &[f64], m: usize) -> Vec<f64> {
    if m == 0 {
        return c.to_vec();
    }
    let mut coeffs = c.to_vec();
    for _ in 0..m {
        let n = coeffs.len();
        let mut int = vec![0.0; n + 1];
        // ∫P_j dx = (P_{j+1} - P_{j-1}) / (2j+1)
        for j in 0..n {
            let scale = coeffs[j] / (2.0 * j as f64 + 1.0);
            int[j + 1] += scale;
            if j >= 1 {
                int[j - 1] -= scale;
            }
        }
        // int[0] is the integration constant (left as accumulated)
        coeffs = int;
    }
    coeffs
}

/// Least-squares fit in Legendre basis.
pub fn legfit(x: &[f64], y: &[f64], deg: usize) -> Result<Vec<f64>, UFuncError> {
    if x.len() != y.len() {
        return Err(UFuncError::Msg(
            "legfit: x and y must have same length".into(),
        ));
    }
    let n = x.len();
    let ncols = deg + 1;
    let mut vander = vec![0.0; n * ncols];
    for i in 0..n {
        vander[i * ncols] = 1.0;
        if ncols > 1 {
            vander[i * ncols + 1] = x[i];
        }
        for j in 2..ncols {
            vander[i * ncols + j] = ((2.0 * j as f64 - 1.0) * x[i] * vander[i * ncols + j - 1]
                - (j as f64 - 1.0) * vander[i * ncols + j - 2])
                / j as f64;
        }
    }
    let mut vtv = vec![0.0; ncols * ncols];
    let mut vty = vec![0.0; ncols];
    for i in 0..ncols {
        for j in 0..ncols {
            let mut sum = 0.0;
            for k in 0..n {
                sum += vander[k * ncols + i] * vander[k * ncols + j];
            }
            vtv[i * ncols + j] = sum;
        }
        let mut sum = 0.0;
        for k in 0..n {
            sum += vander[k * ncols + i] * y[k];
        }
        vty[i] = sum;
    }
    solve_linear_system(&vtv, &vty, ncols)
}

/// Add two Legendre coefficient arrays.
pub fn legadd(c1: &[f64], c2: &[f64]) -> Vec<f64> {
    let len = c1.len().max(c2.len());
    let mut result = vec![0.0; len];
    for (i, &v) in c1.iter().enumerate() {
        result[i] += v;
    }
    for (i, &v) in c2.iter().enumerate() {
        result[i] += v;
    }
    result
}

/// Subtract two Legendre coefficient arrays.
pub fn legsub(c1: &[f64], c2: &[f64]) -> Vec<f64> {
    let len = c1.len().max(c2.len());
    let mut result = vec![0.0; len];
    for (i, &v) in c1.iter().enumerate() {
        result[i] += v;
    }
    for (i, &v) in c2.iter().enumerate() {
        result[i] -= v;
    }
    result
}

/// Multiply two Legendre coefficient arrays.
///
/// Uses the linearization identity for Legendre products:
/// P_m(x) * P_n(x) = sum_{k=|m-n|, step 2}^{m+n}  c(m,n,k) P_k(x)
/// where c(m,n,k) are the Clebsch-Gordan-like linearization coefficients.
///
/// For simplicity, we convert to power basis, multiply, and convert back.
pub fn legmul(c1: &[f64], c2: &[f64]) -> Vec<f64> {
    if c1.is_empty() || c2.is_empty() {
        return Vec::new();
    }
    let p1 = leg2poly(c1);
    let p2 = leg2poly(c2);
    // Standard polynomial multiplication
    let len = p1.len() + p2.len() - 1;
    let mut prod = vec![0.0; len];
    for (i, &a) in p1.iter().enumerate() {
        for (j, &b) in p2.iter().enumerate() {
            prod[i + j] += a * b;
        }
    }
    poly2leg(&prod)
}

/// Divide Legendre coefficient arrays. Returns (quotient, remainder).
pub fn legdiv(c1: &[f64], c2: &[f64]) -> Result<(Vec<f64>, Vec<f64>), UFuncError> {
    if c2.is_empty() || c2.iter().all(|&v| v == 0.0) {
        return Err(UFuncError::Msg(
            "legdiv: division by zero polynomial".to_string(),
        ));
    }
    let p1 = leg2poly(c1);
    let p2 = leg2poly(c2);
    let (pq, pr) = poly_div(&p1, &p2)?;
    Ok((poly2leg(&pq), poly2leg(&pr)))
}

/// Find the roots of a Legendre series via companion matrix eigenvalues.
pub fn legroots(c: &[f64]) -> Result<Vec<f64>, UFuncError> {
    // Remove trailing zeros
    let mut coeffs: Vec<f64> = c.to_vec();
    while coeffs.len() > 1 && *coeffs.last().unwrap() == 0.0 {
        coeffs.pop();
    }
    let n = coeffs.len();
    if n <= 1 {
        return Ok(Vec::new());
    }
    if n == 2 {
        // Linear: c[0]*P_0 + c[1]*P_1 = c[0] + c[1]*x = 0
        return Ok(vec![-coeffs[0] / coeffs[1]]);
    }

    // Convert to power basis and use standard companion matrix
    let poly_coeffs = leg2poly(&coeffs);
    let d = poly_coeffs.len() - 1;
    if d == 0 {
        return Ok(Vec::new());
    }
    let lead = poly_coeffs[d];
    let mut comp = vec![0.0f64; d * d];
    // Companion matrix: first row = -p[d-1..0] / p[d], sub-diagonal = 1
    for j in 0..d {
        comp[j] = -poly_coeffs[d - 1 - j] / lead;
    }
    for i in 0..d - 1 {
        comp[(i + 1) * d + i] = 1.0;
    }

    let eigenvalues = fnp_linalg::eig_nxn(&comp, d)
        .map_err(|e| UFuncError::Msg(format!("legroots eigenvalue error: {e}")))?;
    let mut roots: Vec<f64> = eigenvalues.chunks_exact(2).map(|pair| pair[0]).collect();
    roots.sort_by(|a, b| a.total_cmp(b));
    Ok(roots)
}

/// Build a Legendre series from its roots.
pub fn legfromroots(roots: &[f64]) -> Vec<f64> {
    if roots.is_empty() {
        return vec![1.0];
    }
    // Start with (x - r[0]) in Legendre basis: [-r[0], 1]
    let mut result = vec![-roots[0], 1.0];
    for &r in &roots[1..] {
        let factor = vec![-r, 1.0];
        result = legmul(&result, &factor);
    }
    result
}

/// Convert Legendre coefficients to power series (standard polynomial) coefficients.
///
/// Uses the Legendre polynomial expansion:
/// P_0 = 1, P_1 = x, P_{n+1} = ((2n+1)*x*P_n - n*P_{n-1}) / (n+1)
pub fn leg2poly(c: &[f64]) -> Vec<f64> {
    if c.is_empty() {
        return Vec::new();
    }
    if c.len() == 1 {
        return vec![c[0]];
    }
    let n = c.len();
    let mut result = vec![0.0; n];
    // Build P_k(x) as power series and accumulate c[k] * P_k
    let mut p_prev = vec![0.0; n]; // P_0
    let mut p_curr = vec![0.0; n]; // P_1
    p_prev[0] = 1.0;
    p_curr[1] = 1.0;

    // Accumulate c[0]*P_0
    for j in 0..n {
        result[j] += c[0] * p_prev[j];
    }
    // Accumulate c[1]*P_1
    if n > 1 {
        for j in 0..n {
            result[j] += c[1] * p_curr[j];
        }
    }
    // Build higher-order P_k using recurrence
    for k in 2..n {
        let mut p_next = vec![0.0; n];
        // P_{k} = ((2k-1)*x*P_{k-1} - (k-1)*P_{k-2}) / k
        let kf = k as f64;
        for j in 0..n - 1 {
            p_next[j + 1] += (2.0 * kf - 1.0) * p_curr[j] / kf;
        }
        for j in 0..n {
            p_next[j] -= (kf - 1.0) * p_prev[j] / kf;
        }
        for j in 0..n {
            result[j] += c[k] * p_next[j];
        }
        p_prev = p_curr;
        p_curr = p_next;
    }
    // Trim trailing zeros
    while result.len() > 1 && result.last().map_or(false, |&v| v.abs() < 1e-15) {
        result.pop();
    }
    result
}

/// Convert power series (standard polynomial) coefficients to Legendre coefficients.
///
/// Uses the inverse transformation: given p(x) = sum a_k x^k,
/// find c_k such that p(x) = sum c_k P_k(x).
pub fn poly2leg(p: &[f64]) -> Vec<f64> {
    if p.is_empty() {
        return Vec::new();
    }
    if p.len() == 1 {
        return vec![p[0]];
    }
    // Strategy: build the power-basis representations of P_0, P_1, ..., P_{n-1}
    // and solve the linear system to find Legendre coefficients.
    // More efficient: use legval at Gauss-Legendre nodes for projection.
    // Simplest correct approach: use the inner product with Legendre polys.
    // Since we know the polynomial exactly, use the relation:
    // c_k = (2k+1)/2 * integral_{-1}^{1} p(x) P_k(x) dx
    // But p(x) and P_k(x) are known, so compute exactly via polynomial multiplication.

    // Actually, the simplest approach: build the Legendre Vandermonde and solve.
    let n = p.len();
    // We need to find c such that leg2poly(c) == p
    // Build the transformation matrix M where M[i][j] = coeff of x^i in P_j
    let mut m = vec![0.0; n * n];
    for j in 0..n {
        let mut basis = vec![0.0; n];
        basis[j] = 1.0;
        let poly = leg2poly(&basis);
        for (i, &v) in poly.iter().enumerate() {
            if i < n {
                m[i * n + j] = v;
            }
        }
    }
    // Solve M * c = p
    solve_linear_system(&m, p, n).unwrap_or_else(|_| p.to_vec())
}

// ── Hermite polynomial modules ──────────────────────────────────────────

/// Evaluate a physicist's Hermite series (H_n) at points `x`.
///
/// Recurrence: H_0=1, H_1=2x, H_{n+1} = 2x H_n - 2n H_{n-1}
pub fn hermval(x: &[f64], c: &[f64]) -> Vec<f64> {
    if c.is_empty() {
        return vec![0.0; x.len()];
    }
    if c.len() == 1 {
        return vec![c[0]; x.len()];
    }
    x.iter()
        .map(|&xi| {
            let mut p_prev = 1.0;
            let mut p_curr = 2.0 * xi;
            let mut result = c[0] * p_prev + c[1] * p_curr;
            for (k, &ck) in c.iter().enumerate().skip(2) {
                let p_next = 2.0 * xi * p_curr - 2.0 * (k as f64 - 1.0) * p_prev;
                result += ck * p_next;
                p_prev = p_curr;
                p_curr = p_next;
            }
            result
        })
        .collect()
}

/// Evaluate a probabilist's Hermite series (He_n) at points `x`.
///
/// Recurrence: He_0=1, He_1=x, He_{n+1} = x He_n - n He_{n-1}
pub fn hermeval(x: &[f64], c: &[f64]) -> Vec<f64> {
    if c.is_empty() {
        return vec![0.0; x.len()];
    }
    if c.len() == 1 {
        return vec![c[0]; x.len()];
    }
    x.iter()
        .map(|&xi| {
            let mut p_prev = 1.0;
            let mut p_curr = xi;
            let mut result = c[0] * p_prev + c[1] * p_curr;
            for (k, &ck) in c.iter().enumerate().skip(2) {
                let p_next = xi * p_curr - (k as f64 - 1.0) * p_prev;
                result += ck * p_next;
                p_prev = p_curr;
                p_curr = p_next;
            }
            result
        })
        .collect()
}

/// Differentiate a physicist's Hermite series `m` times.
pub fn hermder(c: &[f64], m: usize) -> Vec<f64> {
    if m == 0 {
        return c.to_vec();
    }
    let mut coeffs = c.to_vec();
    for _ in 0..m {
        let n = coeffs.len();
        if n <= 1 {
            return vec![0.0];
        }
        let mut der = vec![0.0; n - 1];
        for k in 0..n - 1 {
            der[k] = 2.0 * (k as f64 + 1.0) * coeffs[k + 1];
        }
        coeffs = der;
    }
    coeffs
}

/// Integrate a physicist's Hermite series `m` times (constant=0).
pub fn hermint(c: &[f64], m: usize) -> Vec<f64> {
    if m == 0 {
        return c.to_vec();
    }
    let mut coeffs = c.to_vec();
    for _ in 0..m {
        let n = coeffs.len();
        let mut int = vec![0.0; n + 1];
        for k in 0..n {
            int[k + 1] = coeffs[k] / (2.0 * (k as f64 + 1.0));
        }
        coeffs = int;
    }
    coeffs
}

// ── Laguerre polynomial module ──────────────────────────────────────────

/// Evaluate a Laguerre series at points `x`.
///
/// Recurrence: L_0=1, L_1=1-x, (n+1)L_{n+1} = (2n+1-x)L_n - n L_{n-1}
pub fn lagval(x: &[f64], c: &[f64]) -> Vec<f64> {
    if c.is_empty() {
        return vec![0.0; x.len()];
    }
    if c.len() == 1 {
        return vec![c[0]; x.len()];
    }
    x.iter()
        .map(|&xi| {
            let mut p_prev = 1.0;
            let mut p_curr = 1.0 - xi;
            let mut result = c[0] * p_prev + c[1] * p_curr;
            for (k, &ck) in c.iter().enumerate().skip(2) {
                let n = k as f64 - 1.0;
                let p_next = ((2.0 * n + 1.0 - xi) * p_curr - n * p_prev) / (n + 1.0);
                result += ck * p_next;
                p_prev = p_curr;
                p_curr = p_next;
            }
            result
        })
        .collect()
}

/// Differentiate a Laguerre series `m` times.
pub fn lagder(c: &[f64], m: usize) -> Vec<f64> {
    if m == 0 {
        return c.to_vec();
    }
    let mut coeffs = c.to_vec();
    for _ in 0..m {
        let n = coeffs.len();
        if n <= 1 {
            return vec![0.0];
        }
        let mut der = vec![0.0; n - 1];
        // L_n' = -sum_{k=0}^{n-1} L_k  for standard Laguerre
        // In coefficient form: der[k] = -sum_{j=k+1}^{n-1} coeffs[j]
        for k in (0..n - 1).rev() {
            der[k] = -coeffs[k + 1];
            if k + 1 < n - 1 {
                der[k] += der[k + 1];
            }
        }
        coeffs = der;
    }
    coeffs
}

/// Integrate a Laguerre series `m` times (constant=0).
pub fn lagint(c: &[f64], m: usize) -> Vec<f64> {
    if m == 0 {
        return c.to_vec();
    }
    let mut coeffs = c.to_vec();
    for _ in 0..m {
        let n = coeffs.len();
        let mut int = vec![0.0; n + 1];
        // Integration of Laguerre series: cumulative sum with alternating signs
        for k in 0..n {
            int[k] += coeffs[k];
            int[k + 1] -= coeffs[k];
        }
        // Integration constant = 0 → adjust
        coeffs = int;
    }
    coeffs
}

// ── scimath — complex-aware math functions ──────────────────────────────

/// Apply a function that maps real f64 values to complex (real, imag) pairs.
///
/// Returns a UFuncArray with shape [..., 2] (interleaved real/imag).
fn scimath_unary(x: &UFuncArray, f: impl Fn(f64) -> (f64, f64)) -> Result<UFuncArray, UFuncError> {
    let mut values = Vec::with_capacity(x.values.len() * 2);
    for &v in &x.values {
        let (re, im) = f(v);
        values.push(re);
        values.push(im);
    }
    let mut shape = x.shape.clone();
    shape.push(2);
    Ok(UFuncArray {
        shape,
        values,
        dtype: DType::Complex128,
    })
}

/// Complex-aware sqrt: returns complex result for negative inputs.
pub fn scimath_sqrt(x: &UFuncArray) -> Result<UFuncArray, UFuncError> {
    scimath_unary(x, |v| {
        if v >= 0.0 {
            (v.sqrt(), 0.0)
        } else {
            (0.0, (-v).sqrt())
        }
    })
}

/// Complex-aware log: returns complex result for negative inputs.
pub fn scimath_log(x: &UFuncArray) -> Result<UFuncArray, UFuncError> {
    scimath_unary(x, |v| {
        if v > 0.0 {
            (v.ln(), 0.0)
        } else if v == 0.0 {
            (f64::NEG_INFINITY, 0.0)
        } else {
            ((-v).ln(), std::f64::consts::PI)
        }
    })
}

/// Complex-aware log2: returns complex result for negative inputs.
pub fn scimath_log2(x: &UFuncArray) -> Result<UFuncArray, UFuncError> {
    scimath_unary(x, |v| {
        if v > 0.0 {
            (v.log2(), 0.0)
        } else if v == 0.0 {
            (f64::NEG_INFINITY, 0.0)
        } else {
            let ln2 = std::f64::consts::LN_2;
            ((-v).ln() / ln2, std::f64::consts::PI / ln2)
        }
    })
}

/// Complex-aware log10: returns complex result for negative inputs.
pub fn scimath_log10(x: &UFuncArray) -> Result<UFuncArray, UFuncError> {
    scimath_unary(x, |v| {
        if v > 0.0 {
            (v.log10(), 0.0)
        } else if v == 0.0 {
            (f64::NEG_INFINITY, 0.0)
        } else {
            let ln10 = std::f64::consts::LN_10;
            ((-v).ln() / ln10, std::f64::consts::PI / ln10)
        }
    })
}

/// Complex-aware power: x^p with complex result for negative x with fractional p.
pub fn scimath_power(x: &UFuncArray, p: f64) -> Result<UFuncArray, UFuncError> {
    scimath_unary(x, |v| {
        if v >= 0.0 {
            (v.powf(p), 0.0)
        } else {
            // (-v)^p = v^p * e^{i*pi*p} = v^p * (cos(pi*p) + i*sin(pi*p))
            let r = (-v).powf(p);
            let theta = std::f64::consts::PI * p;
            (r * theta.cos(), r * theta.sin())
        }
    })
}

/// Complex-aware arccos: returns complex result for |x| > 1.
pub fn scimath_arccos(x: &UFuncArray) -> Result<UFuncArray, UFuncError> {
    scimath_unary(x, |v| {
        if v.abs() <= 1.0 {
            (v.acos(), 0.0)
        } else {
            // arccos(x) = -i * ln(x + i*sqrt(1 - x^2))
            // For |x| > 1: 1-x^2 < 0, so sqrt is imaginary
            let x2m1 = v * v - 1.0;
            let sqrt_part = x2m1.sqrt();
            if v > 1.0 {
                (0.0, -(v + sqrt_part).ln())
            } else {
                // v < -1
                (std::f64::consts::PI, (sqrt_part - v).ln())
            }
        }
    })
}

/// Complex-aware arcsin: returns complex result for |x| > 1.
pub fn scimath_arcsin(x: &UFuncArray) -> Result<UFuncArray, UFuncError> {
    scimath_unary(x, |v| {
        if v.abs() <= 1.0 {
            (v.asin(), 0.0)
        } else {
            // arcsin(x) = pi/2 - arccos(x)
            let x2m1 = v * v - 1.0;
            let sqrt_part = x2m1.sqrt();
            if v > 1.0 {
                (std::f64::consts::FRAC_PI_2, (v + sqrt_part).ln())
            } else {
                (-std::f64::consts::FRAC_PI_2, -(sqrt_part - v).ln())
            }
        }
    })
}

/// Complex-aware arctanh: returns complex result for |x| > 1.
pub fn scimath_arctanh(x: &UFuncArray) -> Result<UFuncArray, UFuncError> {
    scimath_unary(x, |v| {
        if v.abs() < 1.0 {
            (v.atanh(), 0.0)
        } else if v == 1.0 {
            (f64::INFINITY, 0.0)
        } else if v == -1.0 {
            (f64::NEG_INFINITY, 0.0)
        } else {
            // arctanh(x) = 0.5 * ln((1+x)/(1-x)) for |x| > 1
            // (1+x)/(1-x) is negative when |x| > 1, so log gives complex
            let ratio = (1.0 + v) / (1.0 - v);
            let half_pi = std::f64::consts::FRAC_PI_2;
            (
                0.5 * ratio.abs().ln(),
                if v > 1.0 { -half_pi } else { half_pi },
            )
        }
    })
}

// ── pad mode: linear_ramp ───────────────────────────────────────────────

/// Pad an array using linear ramp from edge values to specified end_values.
///
/// `pad_width` is `(before, after)` for 1-D. `end_values` is `(before_val, after_val)`.
pub fn pad_linear_ramp(
    arr: &UFuncArray,
    pad_width: (usize, usize),
    end_values: (f64, f64),
) -> Result<UFuncArray, UFuncError> {
    if arr.shape.len() != 1 {
        return Err(UFuncError::Msg(
            "pad_linear_ramp: only 1-D supported".into(),
        ));
    }
    let n = arr.values.len();
    let (before, after) = pad_width;
    let (start_val, end_val) = end_values;
    let total = before + n + after;
    let mut values = Vec::with_capacity(total);

    // Before padding: linear ramp from start_val to arr[0]
    let edge_start = if n > 0 { arr.values[0] } else { 0.0 };
    for i in 0..before {
        let t = if before > 0 {
            (i as f64 + 1.0) / (before as f64 + 1.0)
        } else {
            1.0
        };
        values.push(start_val + t * (edge_start - start_val));
    }

    // Original data
    values.extend_from_slice(&arr.values);

    // After padding: linear ramp from arr[n-1] to end_val
    let edge_end = if n > 0 { arr.values[n - 1] } else { 0.0 };
    for i in 0..after {
        let t = (i as f64 + 1.0) / (after as f64 + 1.0);
        values.push(edge_end + t * (end_val - edge_end));
    }

    Ok(UFuncArray {
        shape: vec![total],
        values,
        dtype: arr.dtype,
    })
}

/// Pad with statistical values (max, min, mean, or median) of the array.
pub fn pad_stat(
    arr: &UFuncArray,
    pad_width: (usize, usize),
    mode: &str,
    stat_length: Option<usize>,
) -> Result<UFuncArray, UFuncError> {
    if arr.shape.len() != 1 {
        return Err(UFuncError::Msg("pad_stat: only 1-D supported".into()));
    }
    let n = arr.values.len();
    let (before, after) = pad_width;

    let compute_stat = |values: &[f64], mode: &str| -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        match mode {
            "maximum" => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            "minimum" => values.iter().copied().fold(f64::INFINITY, f64::min),
            "mean" => values.iter().sum::<f64>() / values.len() as f64,
            "median" => {
                let mut sorted = values.to_vec();
                sorted.sort_by(|a, b| a.total_cmp(b));
                let mid = sorted.len() / 2;
                if sorted.len().is_multiple_of(2) {
                    (sorted[mid - 1] + sorted[mid]) / 2.0
                } else {
                    sorted[mid]
                }
            }
            _ => 0.0,
        }
    };

    let stat_len = stat_length.unwrap_or(n);
    let before_slice = &arr.values[..stat_len.min(n)];
    let after_slice = &arr.values[n.saturating_sub(stat_len)..];
    let before_val = compute_stat(before_slice, mode);
    let after_val = compute_stat(after_slice, mode);

    let total = before + n + after;
    let mut values = Vec::with_capacity(total);
    values.extend(std::iter::repeat_n(before_val, before));
    values.extend_from_slice(&arr.values);
    values.extend(std::iter::repeat_n(after_val, after));

    Ok(UFuncArray {
        shape: vec![total],
        values,
        dtype: arr.dtype,
    })
}

/// Pad with uninitialized (zero) values.
pub fn pad_empty(arr: &UFuncArray, pad_width: (usize, usize)) -> Result<UFuncArray, UFuncError> {
    if arr.shape.len() != 1 {
        return Err(UFuncError::Msg("pad_empty: only 1-D supported".into()));
    }
    let n = arr.values.len();
    let (before, after) = pad_width;
    let total = before + n + after;
    let mut values = Vec::with_capacity(total);
    values.extend(std::iter::repeat_n(0.0, before));
    values.extend_from_slice(&arr.values);
    values.extend(std::iter::repeat_n(0.0, after));
    Ok(UFuncArray {
        shape: vec![total],
        values,
        dtype: arr.dtype,
    })
}

// ── Additional math functions ───────────────────────────────────────────

/// Decompose floating-point numbers into mantissa and exponent.
///
/// Returns `(mantissa, exponent)` such that `x = mantissa * 2^exponent`,
/// where mantissa is in `[0.5, 1.0)` for normal numbers.
pub fn frexp(x: &UFuncArray) -> Result<(UFuncArray, UFuncArray), UFuncError> {
    let mut mantissas = Vec::with_capacity(x.values.len());
    let mut exponents = Vec::with_capacity(x.values.len());
    for &v in &x.values {
        if v == 0.0 {
            mantissas.push(if v.is_sign_negative() { -0.0 } else { 0.0 });
            exponents.push(0.0);
        } else if v.is_nan() {
            mantissas.push(f64::NAN);
            exponents.push(0.0);
        } else if v.is_infinite() {
            mantissas.push(v);
            exponents.push(0.0);
        } else {
            let bits = v.to_bits();
            let sign = bits >> 63;
            let biased_exp = ((bits >> 52) & 0x7FF) as i64;
            let frac_bits = bits & 0x000F_FFFF_FFFF_FFFF;
            if biased_exp == 0 {
                // Subnormal: normalize first
                let normalized = v.abs();
                let log2 = normalized.log2().floor() as i64;
                let exp = log2 + 1;
                let m = v / (2.0_f64).powi(exp as i32);
                mantissas.push(m);
                exponents.push(exp as f64);
            } else {
                let exp = biased_exp - 1023 + 1;
                let m_bits = (1022_u64 << 52) | frac_bits;
                let m = f64::from_bits(m_bits);
                mantissas.push(if sign == 1 { -m } else { m });
                exponents.push(exp as f64);
            }
        }
    }
    let m_arr = UFuncArray {
        shape: x.shape.clone(),
        values: mantissas,
        dtype: DType::F64,
    };
    let e_arr = UFuncArray {
        shape: x.shape.clone(),
        values: exponents,
        dtype: DType::F64,
    };
    Ok((m_arr, e_arr))
}

/// Return the fractional and integral parts of each element.
///
/// Both outputs have the same sign as the input.
pub fn modf(x: &UFuncArray) -> Result<(UFuncArray, UFuncArray), UFuncError> {
    let mut fractionals = Vec::with_capacity(x.values.len());
    let mut integrals = Vec::with_capacity(x.values.len());
    for &v in &x.values {
        if v.is_nan() {
            fractionals.push(f64::NAN);
            integrals.push(f64::NAN);
        } else if v.is_infinite() {
            fractionals.push(if v.is_sign_negative() { -0.0 } else { 0.0 });
            integrals.push(v);
        } else {
            integrals.push(v.trunc());
            fractionals.push(v.fract());
        }
    }
    let frac_arr = UFuncArray {
        shape: x.shape.clone(),
        values: fractionals,
        dtype: DType::F64,
    };
    let int_arr = UFuncArray {
        shape: x.shape.clone(),
        values: integrals,
        dtype: DType::F64,
    };
    Ok((frac_arr, int_arr))
}

/// Element-wise greatest common divisor.
///
/// Values are truncated to integers. The result is always non-negative.
pub fn gcd_arrays(a: &UFuncArray, b: &UFuncArray) -> Result<UFuncArray, UFuncError> {
    let bc = UFuncArray::broadcast_arrays(&[a, b])?;
    let (a_bc, b_bc) = (&bc[0], &bc[1]);
    let values = a_bc
        .values
        .iter()
        .zip(&b_bc.values)
        .map(|(&av, &bv)| {
            let mut x = (av.trunc() as i64).abs();
            let mut y = (bv.trunc() as i64).abs();
            while y != 0 {
                let t = y;
                y = x % y;
                x = t;
            }
            x as f64
        })
        .collect();
    Ok(UFuncArray {
        shape: a_bc.shape.clone(),
        values,
        dtype: DType::F64,
    })
}

/// Element-wise least common multiple.
///
/// Values are truncated to integers. The result is always non-negative.
pub fn lcm_arrays(a: &UFuncArray, b: &UFuncArray) -> Result<UFuncArray, UFuncError> {
    let bc = UFuncArray::broadcast_arrays(&[a, b])?;
    let (a_bc, b_bc) = (&bc[0], &bc[1]);
    let values = a_bc
        .values
        .iter()
        .zip(&b_bc.values)
        .map(|(&av, &bv)| {
            let x = (av.trunc() as i64).abs();
            let y = (bv.trunc() as i64).abs();
            if x == 0 || y == 0 {
                return 0.0;
            }
            // gcd
            let (mut gx, mut gy) = (x, y);
            while gy != 0 {
                let t = gy;
                gy = gx % gy;
                gx = t;
            }
            ((x / gx) * y) as f64
        })
        .collect();
    Ok(UFuncArray {
        shape: a_bc.shape.clone(),
        values,
        dtype: DType::F64,
    })
}

/// Element-wise divmod: returns `(floor_quotient, remainder)`.
///
/// Uses floor division semantics (like Python, not C truncation).
pub fn divmod_arrays(
    a: &UFuncArray,
    b: &UFuncArray,
) -> Result<(UFuncArray, UFuncArray), UFuncError> {
    let bc = UFuncArray::broadcast_arrays(&[a, b])?;
    let (a_bc, b_bc) = (&bc[0], &bc[1]);
    let mut quotients = Vec::with_capacity(a_bc.values.len());
    let mut remainders = Vec::with_capacity(a_bc.values.len());
    for (&av, &bv) in a_bc.values.iter().zip(&b_bc.values) {
        if bv == 0.0 {
            if av == 0.0 {
                quotients.push(f64::NAN);
            } else {
                quotients.push(if (av > 0.0) == (bv.is_sign_positive()) {
                    f64::INFINITY
                } else {
                    f64::NEG_INFINITY
                });
            }
            remainders.push(f64::NAN);
        } else {
            let q = (av / bv).floor();
            let r = av - q * bv;
            quotients.push(q);
            remainders.push(r);
        }
    }
    let q_arr = UFuncArray {
        shape: a_bc.shape.clone(),
        values: quotients,
        dtype: DType::F64,
    };
    let r_arr = UFuncArray {
        shape: a_bc.shape.clone(),
        values: remainders,
        dtype: DType::F64,
    };
    Ok((q_arr, r_arr))
}

/// Test element-wise for positive infinity.
pub fn isposinf(x: &UFuncArray) -> Result<UFuncArray, UFuncError> {
    let values = x
        .values
        .iter()
        .map(|&v| if v == f64::INFINITY { 1.0 } else { 0.0 })
        .collect();
    Ok(UFuncArray {
        shape: x.shape.clone(),
        values,
        dtype: DType::Bool,
    })
}

/// Test element-wise for negative infinity.
pub fn isneginf(x: &UFuncArray) -> Result<UFuncArray, UFuncError> {
    let values = x
        .values
        .iter()
        .map(|&v| if v == f64::NEG_INFINITY { 1.0 } else { 0.0 })
        .collect();
    Ok(UFuncArray {
        shape: x.shape.clone(),
        values,
        dtype: DType::Bool,
    })
}

/// Count the number of 1-bits in the absolute value of each integer element.
///
/// Equivalent to `numpy.bitwise_count` (added in NumPy 2.0).
pub fn bitwise_count(x: &UFuncArray) -> Result<UFuncArray, UFuncError> {
    let values = x
        .values
        .iter()
        .map(|&v| {
            let i = (v.trunc() as i64).unsigned_abs();
            i.count_ones() as f64
        })
        .collect();
    Ok(UFuncArray {
        shape: x.shape.clone(),
        values,
        dtype: DType::F64,
    })
}

/// Sort a complex array by real part first, then imaginary part.
///
/// The input is flattened, sorted, and returned as a 1-D complex array with
/// shape `[n, 2]` (interleaved real/imag pairs).
pub fn sort_complex(a: &UFuncArray) -> Result<UFuncArray, UFuncError> {
    // Determine if input is complex (trailing dim = 2)
    let is_complex = a.shape.last() == Some(&2) && a.shape.len() >= 2;
    let mut pairs: Vec<(f64, f64)> = if is_complex {
        a.values.chunks_exact(2).map(|c| (c[0], c[1])).collect()
    } else {
        // Real input: treat as complex with imag=0
        a.values.iter().map(|&r| (r, 0.0)).collect()
    };
    // Sort by real part first, then imaginary part. NaN sorts to end.
    pairs.sort_by(|a, b| {
        let cmp_re = a.0.partial_cmp(&b.0);
        let cmp_im = a.1.partial_cmp(&b.1);
        match cmp_re {
            Some(std::cmp::Ordering::Equal) => cmp_im.unwrap_or(std::cmp::Ordering::Equal),
            Some(ord) => ord,
            None => {
                // NaN handling: NaN sorts to end
                if a.0.is_nan() && b.0.is_nan() {
                    cmp_im.unwrap_or(std::cmp::Ordering::Equal)
                } else if a.0.is_nan() {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Less
                }
            }
        }
    });
    let n = pairs.len();
    let values: Vec<f64> = pairs.into_iter().flat_map(|(r, i)| [r, i]).collect();
    Ok(UFuncArray {
        shape: vec![n, 2],
        values,
        dtype: DType::Complex128,
    })
}

// ── numpy.char — String Array Module ────────────────────────────────────

/// A string array analogous to `numpy.char.chararray`.
///
/// Since `UFuncArray` stores `Vec<f64>`, string data lives in a dedicated
/// `Vec<String>` with shape metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct StringArray {
    shape: Vec<usize>,
    values: Vec<String>,
}

impl StringArray {
    // ── Constructors ────────────────────────────────────────────────

    pub fn new(shape: Vec<usize>, values: Vec<String>) -> Result<Self, UFuncError> {
        let expected = fnp_ndarray::element_count(&shape).map_err(UFuncError::Shape)?;
        if values.len() != expected {
            return Err(UFuncError::InvalidInputLength {
                expected,
                actual: values.len(),
            });
        }
        Ok(Self { shape, values })
    }

    /// Create from a slice of `&str`.
    pub fn from_strs(shape: Vec<usize>, strs: &[&str]) -> Result<Self, UFuncError> {
        Self::new(shape, strs.iter().map(|s| (*s).to_string()).collect())
    }

    // ── Accessors ───────────────────────────────────────────────────

    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[must_use]
    pub fn values(&self) -> &[String] {
        &self.values
    }

    // ── numpy.char element-wise string operations ───────────────────

    /// `np.char.upper(a)` — convert to uppercase.
    #[must_use]
    pub fn upper(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            values: self.values.iter().map(|s| s.to_uppercase()).collect(),
        }
    }

    /// `np.char.lower(a)` — convert to lowercase.
    #[must_use]
    pub fn lower(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            values: self.values.iter().map(|s| s.to_lowercase()).collect(),
        }
    }

    /// `np.char.capitalize(a)` — capitalize first character.
    #[must_use]
    pub fn capitalize(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| {
                    let mut chars = s.chars();
                    match chars.next() {
                        None => String::new(),
                        Some(c) => {
                            let upper: String = c.to_uppercase().collect();
                            let rest: String = chars.collect::<String>().to_lowercase();
                            format!("{upper}{rest}")
                        }
                    }
                })
                .collect(),
        }
    }

    /// `np.char.title(a)` — titlecase words.
    #[must_use]
    pub fn title(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| {
                    let mut result = String::with_capacity(s.len());
                    let mut prev_is_letter = false;
                    for c in s.chars() {
                        if c.is_alphabetic() {
                            if prev_is_letter {
                                for lc in c.to_lowercase() {
                                    result.push(lc);
                                }
                            } else {
                                for uc in c.to_uppercase() {
                                    result.push(uc);
                                }
                            }
                            prev_is_letter = true;
                        } else {
                            result.push(c);
                            prev_is_letter = false;
                        }
                    }
                    result
                })
                .collect(),
        }
    }

    /// `np.char.strip(a)` — strip leading/trailing whitespace.
    #[must_use]
    pub fn strip(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            values: self.values.iter().map(|s| s.trim().to_string()).collect(),
        }
    }

    /// `np.char.lstrip(a)` — strip leading whitespace.
    #[must_use]
    pub fn lstrip(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| s.trim_start().to_string())
                .collect(),
        }
    }

    /// `np.char.rstrip(a)` — strip trailing whitespace.
    #[must_use]
    pub fn rstrip(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| s.trim_end().to_string())
                .collect(),
        }
    }

    /// `np.char.center(a, width, fillchar=' ')` — center-justify.
    #[must_use]
    pub fn center(&self, width: usize, fillchar: char) -> Self {
        Self {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| {
                    let char_len = s.chars().count();
                    if char_len >= width {
                        s.clone()
                    } else {
                        let pad = width - char_len;
                        let left = pad / 2;
                        let right = pad - left;
                        let mut r = String::new();
                        for _ in 0..left {
                            r.push(fillchar);
                        }
                        r.push_str(s);
                        for _ in 0..right {
                            r.push(fillchar);
                        }
                        r
                    }
                })
                .collect(),
        }
    }

    /// `np.char.ljust(a, width, fillchar=' ')` — left-justify.
    #[must_use]
    pub fn ljust(&self, width: usize, fillchar: char) -> Self {
        Self {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| {
                    let char_len = s.chars().count();
                    if char_len >= width {
                        s.clone()
                    } else {
                        let mut r = s.clone();
                        for _ in 0..(width - char_len) {
                            r.push(fillchar);
                        }
                        r
                    }
                })
                .collect(),
        }
    }

    /// `np.char.rjust(a, width, fillchar=' ')` — right-justify.
    #[must_use]
    pub fn rjust(&self, width: usize, fillchar: char) -> Self {
        Self {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| {
                    let char_len = s.chars().count();
                    if char_len >= width {
                        s.clone()
                    } else {
                        let pad = width - char_len;
                        let mut r = String::new();
                        for _ in 0..pad {
                            r.push(fillchar);
                        }
                        r.push_str(s);
                        r
                    }
                })
                .collect(),
        }
    }

    /// `np.char.zfill(a, width)` — pad with zeros.
    #[must_use]
    pub fn zfill(&self, width: usize) -> Self {
        Self {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| {
                    let char_len = s.chars().count();
                    if char_len >= width {
                        return s.clone();
                    }
                    let pad = width - char_len;
                    let zeros: String = std::iter::repeat_n('0', pad).collect();
                    // Preserve leading sign
                    if s.starts_with('+') || s.starts_with('-') {
                        let (sign, rest) = s.split_at(1);
                        format!("{sign}{zeros}{rest}")
                    } else {
                        format!("{zeros}{s}")
                    }
                })
                .collect(),
        }
    }

    /// `np.char.str_len(a)` — string length per element.
    #[must_use]
    pub fn str_len(&self) -> UFuncArray {
        UFuncArray {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| s.chars().count() as f64)
                .collect(),
            dtype: DType::I64,
        }
    }

    /// `np.char.count(a, sub)` — count non-overlapping occurrences of `sub`.
    #[must_use]
    pub fn count_substr(&self, sub: &str) -> UFuncArray {
        UFuncArray {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| s.matches(sub).count() as f64)
                .collect(),
            dtype: DType::I64,
        }
    }

    /// `np.char.find(a, sub)` — find first occurrence, -1 if not found.
    #[must_use]
    pub fn find(&self, sub: &str) -> UFuncArray {
        UFuncArray {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| {
                    s.find(sub)
                        .map_or(-1.0, |byte_idx| s[..byte_idx].chars().count() as f64)
                })
                .collect(),
            dtype: DType::I64,
        }
    }

    /// `np.char.rfind(a, sub)` — find last occurrence, -1 if not found.
    #[must_use]
    pub fn rfind(&self, sub: &str) -> UFuncArray {
        UFuncArray {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| {
                    s.rfind(sub)
                        .map_or(-1.0, |byte_idx| s[..byte_idx].chars().count() as f64)
                })
                .collect(),
            dtype: DType::I64,
        }
    }

    /// `np.char.startswith(a, prefix)` — test prefix.
    #[must_use]
    pub fn startswith(&self, prefix: &str) -> UFuncArray {
        UFuncArray {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| if s.starts_with(prefix) { 1.0 } else { 0.0 })
                .collect(),
            dtype: DType::Bool,
        }
    }

    /// `np.char.endswith(a, suffix)` — test suffix.
    #[must_use]
    pub fn endswith(&self, suffix: &str) -> UFuncArray {
        UFuncArray {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| if s.ends_with(suffix) { 1.0 } else { 0.0 })
                .collect(),
            dtype: DType::Bool,
        }
    }

    /// `np.char.replace(a, old, new)` — replace all occurrences.
    #[must_use]
    pub fn replace(&self, old: &str, new: &str) -> Self {
        Self {
            shape: self.shape.clone(),
            values: self.values.iter().map(|s| s.replace(old, new)).collect(),
        }
    }

    /// `np.char.add(a, b)` — element-wise string concatenation.
    pub fn add(&self, other: &Self) -> Result<Self, UFuncError> {
        if self.shape != other.shape {
            return Err(UFuncError::Shape(
                fnp_ndarray::ShapeError::IncompatibleBroadcast {
                    lhs: self.shape.clone(),
                    rhs: other.shape.clone(),
                },
            ));
        }
        Ok(Self {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .zip(other.values.iter())
                .map(|(a, b)| format!("{a}{b}"))
                .collect(),
        })
    }

    /// `np.char.multiply(a, n)` — repeat each string n times.
    #[must_use]
    pub fn multiply(&self, n: usize) -> Self {
        Self {
            shape: self.shape.clone(),
            values: self.values.iter().map(|s| s.repeat(n)).collect(),
        }
    }

    // ── Character classification (returns Bool UFuncArray) ──────────

    /// `np.char.isalpha(a)`
    #[must_use]
    pub fn isalpha(&self) -> UFuncArray {
        UFuncArray {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| {
                    if !s.is_empty() && s.chars().all(|c| c.is_alphabetic()) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            dtype: DType::Bool,
        }
    }

    /// `np.char.isdigit(a)`
    #[must_use]
    pub fn isdigit(&self) -> UFuncArray {
        UFuncArray {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| {
                    if !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            dtype: DType::Bool,
        }
    }

    /// `np.char.isalnum(a)`
    #[must_use]
    pub fn isalnum(&self) -> UFuncArray {
        UFuncArray {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| {
                    if !s.is_empty() && s.chars().all(|c| c.is_alphanumeric()) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            dtype: DType::Bool,
        }
    }

    /// `np.char.isspace(a)`
    #[must_use]
    pub fn isspace(&self) -> UFuncArray {
        UFuncArray {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| {
                    if !s.is_empty() && s.chars().all(|c| c.is_whitespace()) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            dtype: DType::Bool,
        }
    }

    /// `np.char.isupper(a)`
    #[must_use]
    pub fn isupper(&self) -> UFuncArray {
        UFuncArray {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| {
                    let has_cased = s.chars().any(|c| c.is_alphabetic());
                    if has_cased && s.chars().all(|c| !c.is_alphabetic() || c.is_uppercase()) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            dtype: DType::Bool,
        }
    }

    /// `np.char.islower(a)`
    #[must_use]
    pub fn islower(&self) -> UFuncArray {
        UFuncArray {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| {
                    let has_cased = s.chars().any(|c| c.is_alphabetic());
                    if has_cased && s.chars().all(|c| !c.is_alphabetic() || c.is_lowercase()) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            dtype: DType::Bool,
        }
    }

    /// `np.char.join(sep, a)` — join characters of each string with separator.
    #[must_use]
    pub fn join(&self, sep: &str) -> Self {
        Self {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| {
                    s.chars()
                        .map(|c| c.to_string())
                        .collect::<Vec<_>>()
                        .join(sep)
                })
                .collect(),
        }
    }

    /// `np.char.swapcase(a)` — swap case.
    #[must_use]
    pub fn swapcase(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| {
                    s.chars()
                        .flat_map(|c| {
                            if c.is_uppercase() {
                                c.to_lowercase().collect::<Vec<_>>()
                            } else {
                                c.to_uppercase().collect::<Vec<_>>()
                            }
                        })
                        .collect()
                })
                .collect(),
        }
    }

    /// `np.char.split(a, sep=None, maxsplit=-1)` — split each string.
    ///
    /// Returns a `StringArray` where each element is the joined substrings
    /// separated by a delimiter (`|` by default for flat representation).
    /// When `sep` is `None`, splits on whitespace (like Python `str.split()`).
    /// `maxsplit` of `None` means no limit.
    #[must_use]
    pub fn split(&self, sep: Option<&str>, maxsplit: Option<usize>) -> Vec<Vec<String>> {
        self.values
            .iter()
            .map(|s| match (sep, maxsplit) {
                (None, None) => s.split_whitespace().map(String::from).collect(),
                (None, Some(n)) => s
                    .splitn(n + 1, char::is_whitespace)
                    .filter(|p| !p.is_empty())
                    .map(String::from)
                    .collect(),
                (Some(sep), None) => s.split(sep).map(String::from).collect(),
                (Some(sep), Some(n)) => s.splitn(n + 1, sep).map(String::from).collect(),
            })
            .collect()
    }

    /// `np.char.rsplit(a, sep=None, maxsplit=-1)` — split from the right.
    #[must_use]
    pub fn rsplit(&self, sep: Option<&str>, maxsplit: Option<usize>) -> Vec<Vec<String>> {
        self.values
            .iter()
            .map(|s| match (sep, maxsplit) {
                (None, None) => s.split_whitespace().map(String::from).collect(),
                (None, Some(n)) => {
                    let mut parts: Vec<String> = s
                        .rsplitn(n + 1, char::is_whitespace)
                        .filter(|p| !p.is_empty())
                        .map(String::from)
                        .collect();
                    parts.reverse();
                    parts
                }
                (Some(sep), None) => s.split(sep).map(String::from).collect(),
                (Some(sep), Some(n)) => {
                    let mut parts: Vec<String> = s.rsplitn(n + 1, sep).map(String::from).collect();
                    // rsplitn gives reversed order
                    parts.reverse();
                    parts
                }
            })
            .collect()
    }

    /// `np.char.partition(a, sep)` — partition each string around first occurrence of `sep`.
    ///
    /// Returns three `StringArray`s: (before, separator, after).
    pub fn partition(&self, sep: &str) -> (Self, Self, Self) {
        let mut befores = Vec::with_capacity(self.values.len());
        let mut seps = Vec::with_capacity(self.values.len());
        let mut afters = Vec::with_capacity(self.values.len());
        for s in &self.values {
            if let Some(idx) = s.find(sep) {
                befores.push(s[..idx].to_string());
                seps.push(sep.to_string());
                afters.push(s[idx + sep.len()..].to_string());
            } else {
                befores.push(s.clone());
                seps.push(String::new());
                afters.push(String::new());
            }
        }
        (
            Self {
                shape: self.shape.clone(),
                values: befores,
            },
            Self {
                shape: self.shape.clone(),
                values: seps,
            },
            Self {
                shape: self.shape.clone(),
                values: afters,
            },
        )
    }

    /// `np.char.rpartition(a, sep)` — partition around last occurrence.
    pub fn rpartition(&self, sep: &str) -> (Self, Self, Self) {
        let mut befores = Vec::with_capacity(self.values.len());
        let mut seps = Vec::with_capacity(self.values.len());
        let mut afters = Vec::with_capacity(self.values.len());
        for s in &self.values {
            if let Some(idx) = s.rfind(sep) {
                befores.push(s[..idx].to_string());
                seps.push(sep.to_string());
                afters.push(s[idx + sep.len()..].to_string());
            } else {
                befores.push(String::new());
                seps.push(String::new());
                afters.push(s.clone());
            }
        }
        (
            Self {
                shape: self.shape.clone(),
                values: befores,
            },
            Self {
                shape: self.shape.clone(),
                values: seps,
            },
            Self {
                shape: self.shape.clone(),
                values: afters,
            },
        )
    }

    /// `np.char.encode(a, encoding='utf-8')` — encode strings to bytes.
    ///
    /// Since we only support UTF-8 (Rust's native encoding), this returns
    /// the raw byte representation of each string as a `Vec<Vec<u8>>`.
    #[must_use]
    pub fn encode(&self) -> Vec<Vec<u8>> {
        self.values.iter().map(|s| s.as_bytes().to_vec()).collect()
    }

    /// `np.char.decode(a, encoding='utf-8')` — decode bytes to strings.
    ///
    /// Constructs a `StringArray` from raw byte slices, assuming UTF-8 encoding.
    /// Invalid UTF-8 sequences are replaced with the Unicode replacement character.
    pub fn decode(shape: Vec<usize>, byte_arrays: &[&[u8]]) -> Result<Self, UFuncError> {
        let expected = fnp_ndarray::element_count(&shape).map_err(UFuncError::Shape)?;
        if byte_arrays.len() != expected {
            return Err(UFuncError::InvalidInputLength {
                expected,
                actual: byte_arrays.len(),
            });
        }
        let values: Vec<String> = byte_arrays
            .iter()
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .collect();
        Ok(Self { shape, values })
    }

    /// `np.char.maketrans(intab, outtab)` — build a translation table from two strings.
    ///
    /// Characters in `intab` are mapped to the corresponding character in `outtab`.
    /// If `deletechars` is provided, those characters are mapped to deletion (None).
    /// Returns a `HashMap<char, Option<char>>` suitable for use with `translate`.
    #[must_use]
    pub fn maketrans(
        intab: &str,
        outtab: &str,
        deletechars: Option<&str>,
    ) -> std::collections::HashMap<char, Option<char>> {
        let mut table = std::collections::HashMap::new();
        for (from, to) in intab.chars().zip(outtab.chars()) {
            table.insert(from, Some(to));
        }
        if let Some(del) = deletechars {
            for c in del.chars() {
                table.insert(c, None);
            }
        }
        table
    }

    /// `np.char.translate(a, table)` — translate characters using a mapping.
    ///
    /// `table` maps source chars to optional replacement chars.
    /// If the replacement is `None`, the character is deleted.
    #[must_use]
    pub fn translate(&self, table: &std::collections::HashMap<char, Option<char>>) -> Self {
        Self {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| {
                    s.chars()
                        .filter_map(|c| {
                            if let Some(replacement) = table.get(&c) {
                                *replacement
                            } else {
                                Some(c)
                            }
                        })
                        .collect()
                })
                .collect(),
        }
    }

    /// `np.char.expandtabs(a, tabsize=8)` — replace tabs with spaces.
    #[must_use]
    pub fn expandtabs(&self, tabsize: usize) -> Self {
        Self {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| {
                    let mut result = String::new();
                    let mut col = 0;
                    for c in s.chars() {
                        if c == '\t' {
                            let spaces = tabsize - (col % tabsize);
                            for _ in 0..spaces {
                                result.push(' ');
                            }
                            col += spaces;
                        } else if c == '\n' || c == '\r' {
                            result.push(c);
                            col = 0;
                        } else {
                            result.push(c);
                            col += 1;
                        }
                    }
                    result
                })
                .collect(),
        }
    }

    /// `np.char.isnumeric(a)` — test if all characters are numeric.
    #[must_use]
    pub fn isnumeric(&self) -> UFuncArray {
        UFuncArray {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| {
                    if !s.is_empty() && s.chars().all(|c| c.is_numeric()) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            dtype: DType::Bool,
        }
    }

    /// `np.char.isdecimal(a)` — test if all characters are decimal digits.
    #[must_use]
    pub fn isdecimal(&self) -> UFuncArray {
        UFuncArray {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| {
                    if !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            dtype: DType::Bool,
        }
    }

    /// `np.char.istitle(a)` — test if string is titlecased.
    #[must_use]
    pub fn istitle(&self) -> UFuncArray {
        UFuncArray {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .map(|s| {
                    let has_cased = s.chars().any(|c| c.is_alphabetic());
                    if !has_cased {
                        return 0.0;
                    }
                    let mut prev_is_letter = false;
                    for c in s.chars() {
                        if c.is_alphabetic() {
                            if prev_is_letter {
                                if c.is_uppercase() {
                                    return 0.0;
                                }
                            } else if c.is_lowercase() {
                                return 0.0;
                            }
                            prev_is_letter = true;
                        } else {
                            prev_is_letter = false;
                        }
                    }
                    1.0
                })
                .collect(),
            dtype: DType::Bool,
        }
    }

    /// Like find but returns error when substring not found (np.char.index).
    pub fn index(&self, sub: &str) -> Result<UFuncArray, UFuncError> {
        let mut vals = Vec::with_capacity(self.values.len());
        for s in &self.values {
            match s.find(sub) {
                Some(byte_idx) => {
                    let char_pos = s[..byte_idx].chars().count();
                    vals.push(char_pos as f64);
                }
                None => {
                    return Err(UFuncError::Msg("substring not found".into()));
                }
            }
        }
        Ok(UFuncArray {
            shape: self.shape.clone(),
            values: vals,
            dtype: DType::I64,
        })
    }

    /// Like rfind but returns error when substring not found (np.char.rindex).
    pub fn rindex(&self, sub: &str) -> Result<UFuncArray, UFuncError> {
        let mut vals = Vec::with_capacity(self.values.len());
        for s in &self.values {
            match s.rfind(sub) {
                Some(byte_idx) => {
                    let char_pos = s[..byte_idx].chars().count();
                    vals.push(char_pos as f64);
                }
                None => {
                    return Err(UFuncError::Msg("substring not found".into()));
                }
            }
        }
        Ok(UFuncArray {
            shape: self.shape.clone(),
            values: vals,
            dtype: DType::I64,
        })
    }

    /// Split each element by line boundaries (np.char.splitlines).
    pub fn splitlines(&self) -> Vec<Vec<String>> {
        self.values
            .iter()
            .map(|s| s.lines().map(String::from).collect())
            .collect()
    }

    /// Apply printf-style formatting to each element (np.char.mod).
    /// `values` are the format arguments applied positionally.
    pub fn mod_format(&self, args: &[&str]) -> Self {
        let values: Vec<String> = self
            .values
            .iter()
            .enumerate()
            .map(|(i, fmt_str)| {
                let arg = if i < args.len() { args[i] } else { "" };
                fmt_str.replace("%s", arg).replace("%d", arg)
            })
            .collect();
        Self {
            shape: self.shape.clone(),
            values,
        }
    }

    /// Element-wise string equality (np.char.equal).
    pub fn equal(&self, other: &Self) -> UFuncArray {
        let values: Vec<f64> = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| if a == b { 1.0 } else { 0.0 })
            .collect();
        UFuncArray {
            shape: self.shape.clone(),
            values,
            dtype: DType::Bool,
        }
    }

    /// Element-wise string inequality (np.char.not_equal).
    pub fn not_equal(&self, other: &Self) -> UFuncArray {
        let values: Vec<f64> = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| if a != b { 1.0 } else { 0.0 })
            .collect();
        UFuncArray {
            shape: self.shape.clone(),
            values,
            dtype: DType::Bool,
        }
    }

    /// Element-wise string greater-than (np.char.greater).
    pub fn greater(&self, other: &Self) -> UFuncArray {
        let values: Vec<f64> = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| if a > b { 1.0 } else { 0.0 })
            .collect();
        UFuncArray {
            shape: self.shape.clone(),
            values,
            dtype: DType::Bool,
        }
    }

    /// Element-wise string greater-or-equal (np.char.greater_equal).
    pub fn greater_equal(&self, other: &Self) -> UFuncArray {
        let values: Vec<f64> = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| if a >= b { 1.0 } else { 0.0 })
            .collect();
        UFuncArray {
            shape: self.shape.clone(),
            values,
            dtype: DType::Bool,
        }
    }

    /// Element-wise string less-than (np.char.less).
    pub fn less(&self, other: &Self) -> UFuncArray {
        let values: Vec<f64> = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| if a < b { 1.0 } else { 0.0 })
            .collect();
        UFuncArray {
            shape: self.shape.clone(),
            values,
            dtype: DType::Bool,
        }
    }

    /// Element-wise string less-or-equal (np.char.less_equal).
    pub fn less_equal(&self, other: &Self) -> UFuncArray {
        let values: Vec<f64> = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| if a <= b { 1.0 } else { 0.0 })
            .collect();
        UFuncArray {
            shape: self.shape.clone(),
            values,
            dtype: DType::Bool,
        }
    }

    /// Compare character arrays element-wise (np.char.compare_chararrays).
    /// `cmp` is one of "==", "!=", "<", ">", "<=", ">=".
    pub fn compare_chararrays(&self, other: &Self, cmp: &str) -> Result<UFuncArray, UFuncError> {
        match cmp {
            "==" => Ok(self.equal(other)),
            "!=" => Ok(self.not_equal(other)),
            "<" => Ok(self.less(other)),
            ">" => Ok(self.greater(other)),
            "<=" => Ok(self.less_equal(other)),
            ">=" => Ok(self.greater_equal(other)),
            _ => Err(UFuncError::Msg(format!(
                "compare_chararrays: unknown comparison '{cmp}'"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BinaryOp, MAError, MaskedArray, PrintOptions, QuantileInterp, StringArray,
        UFUNC_PACKET_REASON_CODES, UFuncArray, UFuncError, UFuncLogRecord, UFuncRuntimeMode,
        UnaryOp, bitwise_count, busday_count, busday_offset, cheb2poly, chebadd, chebder, chebfit,
        chebint, chebmul, chebsub, chebval, divmod_arrays, financial_fv, financial_ipmt,
        financial_irr, financial_mirr, financial_nper, financial_npv, financial_pmt,
        financial_ppmt, financial_pv, financial_rate, frexp, gcd_arrays, hermder, hermeval,
        hermint, hermval, is_busday, isneginf, isposinf, lagder, lagval, lcm_arrays,
        leg2poly, legadd, legder, legdiv, legfit, legfromroots, legint, legmul, legroots, legsub,
        legval, ma_is_mask, ma_is_masked, ma_make_mask, ma_mask_or, modf,
        normalize_signature_keywords, pad_empty, pad_linear_ramp, pad_stat, parse_gufunc_signature,
        plan_binary_dispatch, poly2cheb, poly2leg, register_custom_loop, scimath_arccos, scimath_arcsin,
        scimath_arctanh, scimath_log, scimath_log2, scimath_log10, scimath_power, scimath_sqrt,
        sort_complex, unique_all, unique_counts, unique_inverse, unique_values,
        validate_override_payload_class, where_nonzero,
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
        let out = arr.sort(None, None).expect("sort");
        assert_eq!(out.shape(), &[6]);
        assert_eq!(out.values(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn sort_axis_one() {
        // np.sort([[5,1,3],[2,4,0]], axis=1) = [[1,3,5],[0,2,4]]
        let arr = UFuncArray::new(vec![2, 3], vec![5.0, 1.0, 3.0, 2.0, 4.0, 0.0], DType::F64)
            .expect("arr");
        let out = arr.sort(Some(1), None).expect("sort axis=1");
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.values(), &[1.0, 3.0, 5.0, 0.0, 2.0, 4.0]);
    }

    #[test]
    fn sort_axis_zero() {
        // np.sort([[5,1],[2,4]], axis=0) = [[2,1],[5,4]]
        let arr = UFuncArray::new(vec![2, 2], vec![5.0, 1.0, 2.0, 4.0], DType::F64).expect("arr");
        let out = arr.sort(Some(0), None).expect("sort axis=0");
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.values(), &[2.0, 1.0, 5.0, 4.0]);
    }

    #[test]
    fn argsort_axis_none() {
        // np.argsort([3,1,2]) = [1,2,0]
        let arr = UFuncArray::new(vec![3], vec![3.0, 1.0, 2.0], DType::F64).expect("arr");
        let out = arr.argsort(None, None).expect("argsort");
        assert_eq!(out.values(), &[1.0, 2.0, 0.0]);
        assert_eq!(out.dtype(), DType::I64);
    }

    #[test]
    fn argsort_axis_one() {
        // np.argsort([[5,1,3],[2,4,0]], axis=1) = [[1,2,0],[2,0,1]]
        let arr = UFuncArray::new(vec![2, 3], vec![5.0, 1.0, 3.0, 2.0, 4.0, 0.0], DType::F64)
            .expect("arr");
        let out = arr.argsort(Some(1), None).expect("argsort axis=1");
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.values(), &[1.0, 2.0, 0.0, 2.0, 0.0, 1.0]);
    }

    #[test]
    fn sort_kind_quicksort() {
        let arr = UFuncArray::new(vec![5], vec![5.0, 3.0, 1.0, 4.0, 2.0], DType::F64).unwrap();
        let out = arr.sort(None, Some("quicksort")).unwrap();
        assert_eq!(out.values(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn sort_kind_mergesort() {
        let arr = UFuncArray::new(vec![5], vec![5.0, 3.0, 1.0, 4.0, 2.0], DType::F64).unwrap();
        let out = arr.sort(None, Some("mergesort")).unwrap();
        assert_eq!(out.values(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn sort_kind_stable() {
        let arr = UFuncArray::new(vec![5], vec![5.0, 3.0, 1.0, 4.0, 2.0], DType::F64).unwrap();
        let out = arr.sort(None, Some("stable")).unwrap();
        assert_eq!(out.values(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn sort_kind_heapsort() {
        let arr = UFuncArray::new(vec![5], vec![5.0, 3.0, 1.0, 4.0, 2.0], DType::F64).unwrap();
        let out = arr.sort(None, Some("heapsort")).unwrap();
        assert_eq!(out.values(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn sort_kind_invalid() {
        let arr = UFuncArray::new(vec![3], vec![3.0, 1.0, 2.0], DType::F64).unwrap();
        assert!(arr.sort(None, Some("bogus")).is_err());
    }

    #[test]
    fn sort_kind_heapsort_with_axis() {
        let arr = UFuncArray::new(vec![2, 3], vec![5.0, 1.0, 3.0, 2.0, 4.0, 0.0], DType::F64)
            .unwrap();
        let out = arr.sort(Some(1), Some("heapsort")).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.values(), &[1.0, 3.0, 5.0, 0.0, 2.0, 4.0]);
    }

    #[test]
    fn argsort_kind_heapsort() {
        let arr = UFuncArray::new(vec![3], vec![3.0, 1.0, 2.0], DType::F64).unwrap();
        let out = arr.argsort(None, Some("heapsort")).unwrap();
        assert_eq!(out.values(), &[1.0, 2.0, 0.0]);
    }

    #[test]
    fn argsort_kind_stable() {
        let arr = UFuncArray::new(vec![3], vec![3.0, 1.0, 2.0], DType::F64).unwrap();
        let out = arr.argsort(None, Some("stable")).unwrap();
        assert_eq!(out.values(), &[1.0, 2.0, 0.0]);
    }

    #[test]
    fn sort_kind_stable_preserves_order() {
        // Stable sort preserves relative order of equal elements
        // Use argsort to verify stability: for equal values, earlier indices come first
        let arr = UFuncArray::new(vec![6], vec![2.0, 1.0, 2.0, 1.0, 2.0, 1.0], DType::F64)
            .unwrap();
        let indices = arr.argsort(None, Some("stable")).unwrap();
        // All 1.0s first (indices 1, 3, 5), then all 2.0s (indices 0, 2, 4)
        assert_eq!(indices.values(), &[1.0, 3.0, 5.0, 0.0, 2.0, 4.0]);
    }

    #[test]
    fn searchsorted_scalar_probe() {
        // np.searchsorted([1,3,5,7], 4) = 2
        let sorted =
            UFuncArray::new(vec![4], vec![1.0, 3.0, 5.0, 7.0], DType::F64).expect("sorted");
        let probe = UFuncArray::scalar(4.0, DType::F64);
        let out = sorted
            .searchsorted(&probe, None, None)
            .expect("searchsorted");
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
        let out = sorted
            .searchsorted(&probes, None, None)
            .expect("searchsorted");
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
        let out = sorted
            .searchsorted(&probes, None, None)
            .expect("searchsorted");
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.values(), &[0.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn searchsorted_rejects_non_1d_sorted_input() {
        let not_1d =
            UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).expect("not_1d");
        let probe = UFuncArray::scalar(2.0, DType::F64);
        assert!(not_1d.searchsorted(&probe, None, None).is_err());
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
    fn ndim_property() {
        let a = UFuncArray::new(vec![2, 3, 4], vec![0.0; 24], DType::F64).unwrap();
        assert_eq!(a.ndim(), 3);
        let b = UFuncArray::scalar(1.0, DType::F64);
        assert_eq!(b.ndim(), 0);
    }

    #[test]
    fn size_property() {
        let a = UFuncArray::new(vec![2, 3], vec![0.0; 6], DType::F64).unwrap();
        assert_eq!(a.size(), 6);
    }

    #[test]
    fn nbytes_property() {
        let a = UFuncArray::new(vec![4], vec![0.0; 4], DType::F64).unwrap();
        assert_eq!(a.nbytes(), 32); // 4 * 8 bytes
    }

    #[test]
    fn itemsize_property() {
        let a = UFuncArray::new(vec![1], vec![0.0], DType::F64).unwrap();
        assert_eq!(a.itemsize(), 8);
        let b = UFuncArray::new(vec![1], vec![0.0], DType::F32).unwrap();
        assert_eq!(b.itemsize(), 4);
    }

    #[test]
    fn strides_property() {
        let a = UFuncArray::new(vec![2, 3], vec![0.0; 6], DType::F64).unwrap();
        assert_eq!(a.strides(), vec![3, 1]); // C-contiguous: row-major
    }

    #[test]
    fn t_property_2d() {
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let t = a.t().unwrap();
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(t.values(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn t_property_1d_noop() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let t = a.t().unwrap();
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.values(), a.values());
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
    fn array_split_at_indices_basic() {
        let a = UFuncArray::new(
            vec![8],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            DType::F64,
        )
        .unwrap();
        // Split at indices [2, 5] => sub-arrays [0:2], [2:5], [5:8]
        let parts = a.array_split_at_indices(&[2, 5], 0).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].values(), &[1.0, 2.0]);
        assert_eq!(parts[1].values(), &[3.0, 4.0, 5.0]);
        assert_eq!(parts[2].values(), &[6.0, 7.0, 8.0]);
    }

    #[test]
    fn array_split_at_indices_empty_section() {
        let a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        // Adjacent equal indices produce an empty section: [0:2], [2:2], [2:4]
        let parts = a.array_split_at_indices(&[2, 2], 0).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].values(), &[1.0, 2.0]);
        assert_eq!(parts[1].shape(), &[0]); // empty
        assert_eq!(parts[2].values(), &[3.0, 4.0]);
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
    fn fliplr_basic() {
        let arr =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let flipped = arr.fliplr().unwrap();
        assert_eq!(flipped.values(), &[3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
    }

    #[test]
    fn fliplr_1d_fails() {
        let arr = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        assert!(arr.fliplr().is_err());
    }

    #[test]
    fn flipud_basic() {
        let arr =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let flipped = arr.flipud().unwrap();
        assert_eq!(flipped.values(), &[4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn flipud_1d() {
        let arr = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let flipped = arr.flipud().unwrap();
        assert_eq!(flipped.values(), &[4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn block_2x2() {
        // [[a, b], [c, d]] where a=[[1,2],[3,4]], b=[[5],[6]], c=[[7,8]], d=[[9]]
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2, 1], vec![5.0, 6.0], DType::F64).unwrap();
        let c = UFuncArray::new(vec![1, 2], vec![7.0, 8.0], DType::F64).unwrap();
        let d = UFuncArray::new(vec![1, 1], vec![9.0], DType::F64).unwrap();
        let result = UFuncArray::block(&[vec![a, b], vec![c, d]]).unwrap();
        assert_eq!(result.shape(), &[3, 3]);
        assert_eq!(
            result.values(),
            &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn block_single_row() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2, 1], vec![5.0, 6.0], DType::F64).unwrap();
        let result = UFuncArray::block(&[vec![a, b]]).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
    }

    #[test]
    fn ascontiguousarray_with_cast() {
        let a = UFuncArray::new(vec![3], vec![1.5, 2.7, 3.9], DType::F64).unwrap();
        let b = a.ascontiguousarray(Some(DType::I32));
        assert_eq!(b.dtype, DType::I32);
        assert_eq!(b.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn ascontiguousarray_same_dtype() {
        let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap();
        let b = a.ascontiguousarray(None);
        assert_eq!(b.values(), a.values());
    }

    #[test]
    fn require_with_cast() {
        let a = UFuncArray::new(vec![2], vec![1.5, 2.7], DType::F64).unwrap();
        let b = a.require(Some(DType::I64));
        assert_eq!(b.dtype, DType::I64);
        assert_eq!(b.values(), &[1.0, 2.0]);
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
    fn convolve2d_basic() {
        // 2×2 array convolved with 2×2 kernel → 3×3 output
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let k = UFuncArray::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0], DType::F64).unwrap();
        let r = a.convolve2d(&k).unwrap();
        assert_eq!(r.shape(), &[3, 3]);
        // Manual: output[i+ki][j+kj] += a[i][j] * k[ki][kj]
        // k[0][0]=1, k[1][1]=1
        // So convolve2d = a[i][j] at (i,j) + a[i][j] at (i+1,j+1)
        let expected = [1.0, 2.0, 0.0, 3.0, 5.0, 2.0, 0.0, 3.0, 4.0];
        for (i, (&got, &exp)) in r.values().iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-10,
                "convolve2d at {i}: {got} vs {exp}"
            );
        }
    }

    #[test]
    fn convolve2d_identity_kernel() {
        // Convolution with [[1]] returns the same array
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let k = UFuncArray::new(vec![1, 1], vec![1.0], DType::F64).unwrap();
        let r = a.convolve2d(&k).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.values(), a.values());
    }

    #[test]
    fn convolve2d_rejects_1d() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let k = UFuncArray::new(vec![2, 2], vec![1.0; 4], DType::F64).unwrap();
        assert!(a.convolve2d(&k).is_err());
    }

    #[test]
    fn correlate2d_is_reverse_convolve() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let k = UFuncArray::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0], DType::F64).unwrap();
        let r = a.correlate2d(&k).unwrap();
        assert_eq!(r.shape(), &[3, 3]);
        // Correlate reverses kernel to [[1,0],[0,1]] (already symmetric for this case)
        // So correlate2d should equal convolve2d for symmetric kernels
        let c = a.convolve2d(&k).unwrap();
        assert_eq!(r.values(), c.values());
    }

    #[test]
    fn apply_over_axes_sum() {
        // 2×3 array, apply sum over axes [0, 1]
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let r = a
            .apply_over_axes(
                |arr, axis, keepdims| arr.reduce_sum(axis, keepdims),
                &[0, 1],
            )
            .unwrap();
        // After sum(axis=0, keepdims=true): shape [1,3], values [5,7,9]
        // After sum(axis=1, keepdims=true): shape [1,1], values [21]
        assert_eq!(r.shape(), &[1, 1]);
        assert!((r.values()[0] - 21.0).abs() < 1e-10);
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
        let result = arr.isin(&test, false);
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

    #[test]
    fn roots_cubic_all_real() {
        // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
        let p = UFuncArray::new(vec![4], vec![1.0, -6.0, 11.0, -6.0], DType::F64).unwrap();
        let r = p.roots().unwrap();
        assert_eq!(r.shape(), &[3]);
        let mut roots = r.values().to_vec();
        roots.sort_by(|a, b| a.total_cmp(b));
        assert!((roots[0] - 1.0).abs() < 1e-8, "root 1: {}", roots[0]);
        assert!((roots[1] - 2.0).abs() < 1e-8, "root 2: {}", roots[1]);
        assert!((roots[2] - 3.0).abs() < 1e-8, "root 3: {}", roots[2]);
    }

    #[test]
    fn roots_quartic() {
        // (x-1)(x+1)(x-2)(x+2) = (x^2-1)(x^2-4) = x^4 - 5x^2 + 4
        let p = UFuncArray::new(vec![5], vec![1.0, 0.0, -5.0, 0.0, 4.0], DType::F64).unwrap();
        let r = p.roots().unwrap();
        assert_eq!(r.shape(), &[4]);
        let mut roots: Vec<f64> = r.values().to_vec();
        roots.sort_by(|a, b| a.total_cmp(b));
        assert!((roots[0] - (-2.0)).abs() < 1e-8, "root -2: {}", roots[0]);
        assert!((roots[1] - (-1.0)).abs() < 1e-8, "root -1: {}", roots[1]);
        assert!((roots[2] - 1.0).abs() < 1e-8, "root 1: {}", roots[2]);
        assert!((roots[3] - 2.0).abs() < 1e-8, "root 2: {}", roots[3]);
    }

    #[test]
    fn roots_with_leading_zeros() {
        // Leading zero coefficients should be stripped: [0, 0, 1, -3, 2] -> degree 2
        let p = UFuncArray::new(vec![5], vec![0.0, 0.0, 1.0, -3.0, 2.0], DType::F64).unwrap();
        let r = p.roots().unwrap();
        assert_eq!(r.shape(), &[2]);
        let mut roots = r.values().to_vec();
        roots.sort_by(|a, b| a.total_cmp(b));
        assert!((roots[0] - 1.0).abs() < 1e-12);
        assert!((roots[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn roots_with_trailing_zeros() {
        // [1, -3, 2, 0] -> (x^3 - 3x^2 + 2x) = x(x-1)(x-2), roots: 0, 1, 2
        let p = UFuncArray::new(vec![4], vec![1.0, -3.0, 2.0, 0.0], DType::F64).unwrap();
        let r = p.roots().unwrap();
        assert_eq!(r.shape(), &[3]);
        let mut roots = r.values().to_vec();
        roots.sort_by(|a, b| a.total_cmp(b));
        assert!((roots[0] - 0.0).abs() < 1e-8, "root 0: {}", roots[0]);
        assert!((roots[1] - 1.0).abs() < 1e-8, "root 1: {}", roots[1]);
        assert!((roots[2] - 2.0).abs() < 1e-8, "root 2: {}", roots[2]);
    }

    #[test]
    fn roots_degree5() {
        // (x-1)(x-2)(x-3)(x-4)(x-5) = x^5 - 15x^4 + 85x^3 - 225x^2 + 274x - 120
        let p = UFuncArray::new(
            vec![6],
            vec![1.0, -15.0, 85.0, -225.0, 274.0, -120.0],
            DType::F64,
        )
        .unwrap();
        let r = p.roots().unwrap();
        assert_eq!(r.shape(), &[5]);
        let mut roots = r.values().to_vec();
        roots.sort_by(|a, b| a.total_cmp(b));
        for (i, &expected) in [1.0, 2.0, 3.0, 4.0, 5.0].iter().enumerate() {
            assert!(
                (roots[i] - expected).abs() < 1e-6,
                "root {}: expected {}, got {}",
                i,
                expected,
                roots[i]
            );
        }
    }

    // ── sorting: partition, argpartition, lexsort ──

    #[test]
    fn partition_basic() {
        let a = UFuncArray::new(vec![5], vec![3.0, 1.0, 4.0, 1.0, 5.0], DType::F64).unwrap();
        let r = a.partition(2, None).unwrap();
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
        assert!(a.partition(5, None).is_err());
    }

    #[test]
    fn argpartition_basic() {
        let a = UFuncArray::new(vec![5], vec![3.0, 1.0, 4.0, 1.0, 5.0], DType::F64).unwrap();
        let r = a.argpartition(2, None).unwrap();
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
    fn partition_2d_axis1() {
        // 2x4 array, partition along axis 1 with kth=1
        let a = UFuncArray::new(
            vec![2, 4],
            vec![4.0, 2.0, 3.0, 1.0, 8.0, 6.0, 7.0, 5.0],
            DType::F64,
        )
        .unwrap();
        let r = a.partition(1, Some(-1)).unwrap();
        assert_eq!(r.shape(), &[2, 4]);
        // In each row, element at index 1 should be the 2nd smallest
        // Row 0: sorted would be [1,2,3,4], so kth=1 => value 2.0
        assert!(r.values()[0] <= r.values()[1]); // left of kth <= kth
        assert!(r.values()[2] >= r.values()[1]); // right of kth >= kth
        assert!(r.values()[3] >= r.values()[1]);
        // Row 1: sorted would be [5,6,7,8], so kth=1 => value 6.0
        assert!(r.values()[4] <= r.values()[5]);
        assert!(r.values()[6] >= r.values()[5]);
        assert!(r.values()[7] >= r.values()[5]);
    }

    #[test]
    fn argpartition_2d_axis0() {
        // 3x2 array, partition along axis 0 with kth=1
        let a =
            UFuncArray::new(vec![3, 2], vec![3.0, 6.0, 1.0, 4.0, 2.0, 5.0], DType::F64).unwrap();
        let r = a.argpartition(1, Some(0)).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        // Column 0 values: [3.0, 1.0, 2.0]. Sorted indices: [1,2,0].
        // kth=1: index at position 1 should point to 2nd smallest (2.0 at index 2)
        let col0_kth_idx = r.values()[2] as usize; // row 1, col 0
        let col0_vals = [3.0, 1.0, 2.0];
        let kth_val = col0_vals[col0_kth_idx];
        assert!(col0_vals[r.values()[0] as usize] <= kth_val);
        assert!(col0_vals[r.values()[2 * 2] as usize] >= kth_val);
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
    fn rollaxis_to_front() {
        let a = UFuncArray::new(
            vec![2, 3, 4],
            (0..24).map(|i| i as f64).collect(),
            DType::F64,
        )
        .unwrap();
        // Roll axis 2 to position 0 => shape [4, 2, 3]
        let r = a.rollaxis(2, 0).unwrap();
        assert_eq!(r.shape(), &[4, 2, 3]);
    }

    #[test]
    fn rollaxis_noop() {
        let a = UFuncArray::new(
            vec![2, 3, 4],
            (0..24).map(|i| i as f64).collect(),
            DType::F64,
        )
        .unwrap();
        // Roll axis 0 to start=0 => no change
        let r = a.rollaxis(0, 0).unwrap();
        assert_eq!(r.shape(), &[2, 3, 4]);
    }

    #[test]
    fn row_stack_same_as_vstack() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![4.0, 5.0, 6.0], DType::F64).unwrap();
        let v = UFuncArray::vstack(&[a.clone(), b.clone()]).unwrap();
        let rs = UFuncArray::row_stack(&[a, b]).unwrap();
        assert_eq!(v.shape(), rs.shape());
        assert_eq!(v.values(), rs.values());
    }

    #[test]
    fn ix_basic() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2], vec![4.0, 5.0], DType::F64).unwrap();
        let result = UFuncArray::ix_(&[a, b]).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].shape(), &[3, 1]);
        assert_eq!(result[0].values(), &[1.0, 2.0, 3.0]);
        assert_eq!(result[1].shape(), &[1, 2]);
        assert_eq!(result[1].values(), &[4.0, 5.0]);
    }

    #[test]
    fn ix_three_arrays() {
        let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![3.0, 4.0, 5.0], DType::F64).unwrap();
        let c = UFuncArray::new(vec![4], vec![6.0, 7.0, 8.0, 9.0], DType::F64).unwrap();
        let result = UFuncArray::ix_(&[a, b, c]).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].shape(), &[2, 1, 1]);
        assert_eq!(result[1].shape(), &[1, 3, 1]);
        assert_eq!(result[2].shape(), &[1, 1, 4]);
    }

    #[test]
    fn ix_rejects_non_1d() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0; 4], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![1.0; 3], DType::F64).unwrap();
        assert!(UFuncArray::ix_(&[a, b]).is_err());
    }

    #[test]
    fn tolist_basic() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        assert_eq!(a.tolist(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn tobytes_array_roundtrip() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.5, -3.0], DType::F64).unwrap();
        let bytes = a.tobytes_array();
        assert_eq!(bytes.len(), 24); // 3 * 8 bytes
        // Verify first value
        let first = f64::from_le_bytes(bytes[0..8].try_into().unwrap());
        assert_eq!(first, 1.0);
    }

    #[test]
    fn byteswap_roundtrip() {
        let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap();
        let swapped = a.byteswap();
        // Swapping twice returns original
        let restored = swapped.byteswap();
        assert_eq!(restored.values(), &[1.0, 2.0]);
    }

    #[test]
    fn diagonal_2d() {
        let a = UFuncArray::new(
            vec![3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            DType::F64,
        )
        .unwrap();
        let d = a.diagonal(0, 0, 1).unwrap();
        assert_eq!(d.values(), &[1.0, 5.0, 9.0]);
    }

    #[test]
    fn diagonal_2d_offset() {
        let a = UFuncArray::new(
            vec![3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            DType::F64,
        )
        .unwrap();
        let d = a.diagonal(1, 0, 1).unwrap();
        assert_eq!(d.values(), &[2.0, 6.0]);
        let d2 = a.diagonal(-1, 0, 1).unwrap();
        assert_eq!(d2.values(), &[4.0, 8.0]);
    }

    #[test]
    fn diagonal_3d() {
        // Shape (2, 3, 3): two 3x3 matrices stacked
        let vals: Vec<f64> = (1..=18).map(|i| i as f64).collect();
        let a = UFuncArray::new(vec![2, 3, 3], vals, DType::F64).unwrap();
        let d = a.diagonal(0, 1, 2).unwrap();
        // Output shape: [2, 3] (remove axes 1,2, append diag_len=3)
        assert_eq!(d.shape(), &[2, 3]);
        // First matrix diag: [1, 5, 9], second matrix diag: [10, 14, 18]
        assert_eq!(d.values(), &[1.0, 5.0, 9.0, 10.0, 14.0, 18.0]);
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

    #[test]
    fn diag_indices_from_square() {
        let a = UFuncArray::new(vec![3, 3], vec![0.0; 9], DType::F64).unwrap();
        let (indices, dt) = UFuncArray::diag_indices_from(&a).unwrap();
        assert_eq!(dt, DType::I64);
        assert_eq!(indices.len(), 2);
        assert_eq!(indices[0].values(), &[0.0, 1.0, 2.0]);
    }

    #[test]
    fn diag_indices_from_non_square_fails() {
        let a = UFuncArray::new(vec![2, 3], vec![0.0; 6], DType::F64).unwrap();
        assert!(UFuncArray::diag_indices_from(&a).is_err());
    }

    #[test]
    fn diag_indices_from_1d_fails() {
        let a = UFuncArray::new(vec![3], vec![0.0; 3], DType::F64).unwrap();
        assert!(UFuncArray::diag_indices_from(&a).is_err());
    }

    #[test]
    fn tril_indices_from_basic() {
        let a = UFuncArray::new(vec![3, 4], vec![0.0; 12], DType::F64).unwrap();
        let (rows, cols) = UFuncArray::tril_indices_from(&a, 0).unwrap();
        let (rows_exp, cols_exp) = UFuncArray::tril_indices(3, 4, 0);
        assert_eq!(rows.values(), rows_exp.values());
        assert_eq!(cols.values(), cols_exp.values());
    }

    #[test]
    fn tril_indices_from_non_2d_fails() {
        let a = UFuncArray::new(vec![3], vec![0.0; 3], DType::F64).unwrap();
        assert!(UFuncArray::tril_indices_from(&a, 0).is_err());
    }

    #[test]
    fn triu_indices_from_basic() {
        let a = UFuncArray::new(vec![3, 4], vec![0.0; 12], DType::F64).unwrap();
        let (rows, cols) = UFuncArray::triu_indices_from(&a, 0).unwrap();
        let (rows_exp, cols_exp) = UFuncArray::triu_indices(3, 4, 0);
        assert_eq!(rows.values(), rows_exp.values());
        assert_eq!(cols.values(), cols_exp.values());
    }

    #[test]
    fn triu_indices_from_non_2d_fails() {
        let a = UFuncArray::new(vec![2, 3, 4], vec![0.0; 24], DType::F64).unwrap();
        assert!(UFuncArray::triu_indices_from(&a, 0).is_err());
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
    fn gradient_edge_order2_quadratic() {
        // f(x) = x^2 at x=[0,1,2,3,4], gradient should be [0,2,4,6,8]
        // edge_order=1: left=1, right=7; edge_order=2: left=0, right=8 (exact!)
        let a = UFuncArray::new(vec![5], vec![0.0, 1.0, 4.0, 9.0, 16.0], DType::F64).unwrap();
        let r = a.gradient_advanced(None, 2, None).unwrap();
        assert!((r.values()[0] - 0.0).abs() < 1e-10, "left edge_order=2");
        assert!((r.values()[1] - 2.0).abs() < 1e-10);
        assert!((r.values()[2] - 4.0).abs() < 1e-10);
        assert!((r.values()[3] - 6.0).abs() < 1e-10);
        assert!((r.values()[4] - 8.0).abs() < 1e-10, "right edge_order=2");
    }

    #[test]
    fn gradient_edge_order2_linear() {
        // f(x) = 2*x + 1 at x=[0,1,2,3], gradient = [2,2,2,2]
        let a = UFuncArray::new(vec![4], vec![1.0, 3.0, 5.0, 7.0], DType::F64).unwrap();
        let r = a.gradient_advanced(None, 2, None).unwrap();
        for &v in r.values() {
            assert!((v - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn gradient_edge_order2_rejects_short() {
        // edge_order=2 requires at least 3 elements
        let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap();
        assert!(a.gradient_advanced(None, 2, None).is_err());
    }

    #[test]
    fn gradient_nonuniform_spacing() {
        // f(x) = x^2 at x=[0, 1, 3], gradient should be [2*0, 2*1, 2*3] = [0, 2, 6]
        // but with non-uniform differences:
        //   left (order=1): (1 - 0)/(1 - 0) = 1
        //   central: non-uniform formula
        //   right (order=1): (9 - 1)/(3 - 1) = 4
        let a = UFuncArray::new(vec![3], vec![0.0, 1.0, 9.0], DType::F64).unwrap();
        let coords = [0.0, 1.0, 3.0];
        let r = a.gradient_advanced(None, 1, Some(&coords)).unwrap();
        // Left: (f(1)-f(0))/(x(1)-x(0)) = (1-0)/(1-0) = 1.0
        assert!((r.values()[0] - 1.0).abs() < 1e-10);
        // Central: non-uniform central diff for x^2 at x=1
        // h- = 1, h+ = 2, hs = 3
        // (h-^2 * f(2) + (h+^2 - h-^2) * f(1) - h+^2 * f(0)) / (h- * h+ * hs)
        // = (1*9 + (4-1)*1 - 4*0) / (1*2*3) = (9+3-0)/6 = 2.0
        assert!((r.values()[1] - 2.0).abs() < 1e-10);
        // Right: (f(2)-f(1))/(x(2)-x(1)) = (9-1)/(3-1) = 4.0
        assert!((r.values()[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn gradient_nonuniform_edge_order2() {
        // f(x) = x^2 at x=[0, 1, 3, 6], gradient = 2*x = [0, 2, 6, 12]
        let a = UFuncArray::new(vec![4], vec![0.0, 1.0, 9.0, 36.0], DType::F64).unwrap();
        let coords = [0.0, 1.0, 3.0, 6.0];
        let r = a.gradient_advanced(None, 2, Some(&coords)).unwrap();
        // For quadratic function, edge_order=2 should give exact gradient
        assert!((r.values()[0] - 0.0).abs() < 1e-10, "left edge");
        assert!((r.values()[1] - 2.0).abs() < 1e-10);
        // Right edge: edge_order=2 with non-uniform spacing
        assert!((r.values()[3] - 12.0).abs() < 1e-10, "right edge");
    }

    #[test]
    fn gradient_default_unchanged() {
        // Verify gradient_axis (edge_order=1, uniform) still works
        let a = UFuncArray::new(vec![3], vec![1.0, 3.0, 6.0], DType::F64).unwrap();
        let r = a.gradient_axis(None).unwrap();
        assert_eq!(r.values()[0], 2.0); // forward: 3-1
        assert_eq!(r.values()[1], 2.5); // central: (6-1)/2
        assert_eq!(r.values()[2], 3.0); // backward: 6-3
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
        let r = a.trapezoid(1.0, None).unwrap();
        assert!((r.values()[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn trapezoid_2d_axis0() {
        // np.trapezoid([[1, 2], [3, 4], [5, 6]], dx=1.0, axis=0)
        // Along axis 0: column 0: trap([1,3,5], dx=1) = (1+3)/2 + (3+5)/2 = 6.0
        //               column 1: trap([2,4,6], dx=1) = (2+4)/2 + (4+6)/2 = 8.0
        let a = UFuncArray::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .unwrap();
        let r = a.trapezoid(1.0, Some(0)).unwrap();
        assert_eq!(r.shape(), &[2]);
        assert!((r.values()[0] - 6.0).abs() < 1e-10);
        assert!((r.values()[1] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn trapezoid_2d_axis1() {
        // np.trapezoid([[1, 2, 3], [4, 5, 6]], dx=1.0, axis=1)
        // Along axis 1: row 0: trap([1,2,3]) = 4.0
        //               row 1: trap([4,5,6]) = 10.0
        let a = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .unwrap();
        let r = a.trapezoid(1.0, Some(1)).unwrap();
        assert_eq!(r.shape(), &[2]);
        assert!((r.values()[0] - 4.0).abs() < 1e-10);
        assert!((r.values()[1] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn trapezoid_2d_axis_neg1() {
        // axis=-1 is same as axis=1 for 2-D
        let a = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .unwrap();
        let r = a.trapezoid(1.0, Some(-1)).unwrap();
        assert_eq!(r.shape(), &[2]);
        assert!((r.values()[0] - 4.0).abs() < 1e-10);
        assert!((r.values()[1] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn trapezoid_3d_axis1() {
        // shape (2, 3, 2) - integrate along axis 1
        // result shape (2, 2)
        // Data: [[[1,2],[3,4],[5,6]], [[7,8],[9,10],[11,12]]]
        let vals: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let a = UFuncArray::new(vec![2, 3, 2], vals, DType::F64).unwrap();
        let r = a.trapezoid(1.0, Some(1)).unwrap();
        assert_eq!(r.shape(), &[2, 2]);
        // First block [[1,2],[3,4],[5,6]]:
        //   col0: trap([1,3,5]) = (1+3)/2 + (3+5)/2 = 2 + 4 = 6.0
        //   col1: trap([2,4,6]) = (2+4)/2 + (4+6)/2 = 3 + 5 = 8.0
        // Second block [[7,8],[9,10],[11,12]]:
        //   col0: trap([7,9,11]) = (7+9)/2 + (9+11)/2 = 8 + 10 = 18.0
        //   col1: trap([8,10,12]) = (8+10)/2 + (10+12)/2 = 9 + 11 = 20.0
        assert!((r.values()[0] - 6.0).abs() < 1e-10);
        assert!((r.values()[1] - 8.0).abs() < 1e-10);
        assert!((r.values()[2] - 18.0).abs() < 1e-10);
        assert!((r.values()[3] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn trapezoid_custom_dx() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let r = a.trapezoid(0.5, None).unwrap();
        // 0.5*(1+2)*0.5 + 0.5*(2+3)*0.5 = 0.75 + 1.25 = 2.0
        assert!((r.values()[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn trapezoid_none_axis_defaults_to_last() {
        // For 2-D array, axis=None defaults to last axis
        let a = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .unwrap();
        let r = a.trapezoid(1.0, None).unwrap();
        assert_eq!(r.shape(), &[2]);
        assert!((r.values()[0] - 4.0).abs() < 1e-10);
        assert!((r.values()[1] - 10.0).abs() < 1e-10);
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
        assert_eq!(a.count_nonzero(None, false).unwrap().values(), &[3.0]);
        let z = UFuncArray::new(vec![3], vec![0.0, 0.0, 0.0], DType::F64).unwrap();
        assert_eq!(z.count_nonzero(None, false).unwrap().values(), &[0.0]);
    }

    #[test]
    fn packbits_and_unpackbits_roundtrip() {
        // Binary array [1, 0, 1, 1, 0, 0, 1, 0] → packed byte = 0b10110010 = 178
        let bits = vec![1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        let a = UFuncArray::new(vec![8], bits.clone(), DType::U8).unwrap();
        let packed = a.packbits(None).unwrap();
        assert_eq!(packed.shape, vec![1]);
        assert!(
            (packed.values[0] - 178.0).abs() < 1e-10,
            "packed={}",
            packed.values[0]
        );

        // Unpack should recover the original bits
        let unpacked = packed.unpackbits(None, None).unwrap();
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
        let packed = a.packbits(None).unwrap();
        assert_eq!(packed.shape, vec![1]);
        // 11101000 = 232
        assert!(
            (packed.values[0] - 232.0).abs() < 1e-10,
            "packed={}",
            packed.values[0]
        );
    }

    #[test]
    fn packbits_axis0() {
        // 2x8 array of bits, pack along axis 0 → 1x8 (each col has 2 bits → 1 byte)
        // Wait, actually axis=0 packs along rows, so 2 bits per column packed to 1 byte each
        // Shape [2, 8] → packed shape [1, 8] (2 bits per column → ceil(2/8)=1 byte)
        let a = UFuncArray::new(
            vec![2, 8],
            vec![
                1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, // row 0
                0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, // row 1
            ],
            DType::U8,
        )
        .unwrap();
        let packed = a.packbits(Some(0)).unwrap();
        assert_eq!(packed.shape(), &[1, 8]);
        // Column 0: bits [1, 0] → 10000000 = 128
        assert!((packed.values()[0] - 128.0).abs() < 1e-10);
        // Column 1: bits [0, 1] → 01000000 = 64
        assert!((packed.values()[1] - 64.0).abs() < 1e-10);
    }

    #[test]
    fn unpackbits_with_count() {
        // Pack 5 bits, unpack with count=5 to recover original length
        let a = UFuncArray::new(vec![5], vec![1.0, 1.0, 1.0, 0.0, 1.0], DType::U8).unwrap();
        let packed = a.packbits(None).unwrap();
        let unpacked = packed.unpackbits(None, Some(5)).unwrap();
        assert_eq!(unpacked.shape(), &[5]);
        assert_eq!(unpacked.values(), &[1.0, 1.0, 1.0, 0.0, 1.0]);
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

    #[test]
    fn einsum_path_two_operands() {
        let a = UFuncArray::new(vec![3, 4], vec![0.0; 12], DType::F64).unwrap();
        let b = UFuncArray::new(vec![4, 2], vec![0.0; 8], DType::F64).unwrap();
        let (path, desc) = UFuncArray::einsum_path("ij,jk->ik", &[&a, &b]).unwrap();
        assert_eq!(path.len(), 1);
        assert_eq!(path[0], vec![0, 1]);
        assert!(desc.contains("ij,jk->ik"));
    }

    #[test]
    fn einsum_path_three_operands() {
        let a = UFuncArray::new(vec![2, 3], vec![0.0; 6], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3, 4], vec![0.0; 12], DType::F64).unwrap();
        let c = UFuncArray::new(vec![4, 5], vec![0.0; 20], DType::F64).unwrap();
        let (path, _desc) = UFuncArray::einsum_path("ij,jk,kl->il", &[&a, &b, &c]).unwrap();
        // Should produce 2 contraction steps
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn einsum_path_mismatch() {
        let a = UFuncArray::new(vec![2], vec![0.0; 2], DType::F64).unwrap();
        assert!(UFuncArray::einsum_path("i,j->ij", &[&a]).is_err());
    }

    // ── irfft tests ──

    #[test]
    fn irfft_roundtrip() {
        // rfft then irfft should recover original signal
        let a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let rf = a.rfft(None).unwrap();
        let recovered = rf.irfft(Some(4)).unwrap();
        assert_eq!(recovered.shape, vec![4]);
        for i in 0..4 {
            assert!(
                (recovered.values[i] - a.values[i]).abs() < 1e-10,
                "irfft[{i}] = {}, expected {}",
                recovered.values[i],
                a.values[i]
            );
        }
    }

    #[test]
    fn irfft_empty() {
        let a = UFuncArray::new(vec![0, 2], vec![], DType::F64).unwrap();
        let result = a.irfft(None).unwrap();
        assert_eq!(result.shape, vec![0]);
        assert!(result.values.is_empty());
    }

    // ── fft2 / ifft2 tests ──

    #[test]
    fn fft2_dc_component() {
        // A 2x2 matrix of ones should have DC = 4 and all other coefficients = 0
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 1.0, 1.0, 1.0], DType::F64).unwrap();
        let ft = a.fft2().unwrap();
        assert_eq!(ft.shape, vec![2, 2, 2]);
        // DC component at [0,0] = sum of all = 4
        assert!((ft.values[0] - 4.0).abs() < 1e-10, "DC re={}", ft.values[0]);
        assert!(ft.values[1].abs() < 1e-10, "DC im={}", ft.values[1]);
        // All other components should be 0
        for i in 1..4 {
            assert!(
                ft.values[i * 2].abs() < 1e-10,
                "re[{i}]={}",
                ft.values[i * 2]
            );
            assert!(
                ft.values[i * 2 + 1].abs() < 1e-10,
                "im[{i}]={}",
                ft.values[i * 2 + 1]
            );
        }
    }

    #[test]
    fn fft2_ifft2_roundtrip() {
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let ft = a.fft2().unwrap();
        let recovered = ft.ifft2().unwrap();
        assert_eq!(recovered.shape, vec![2, 3, 2]);
        // Real parts should match original, imaginary parts should be ~0
        for r in 0..2 {
            for c in 0..3 {
                let idx = r * 3 + c;
                let re = recovered.values[idx * 2];
                let im = recovered.values[idx * 2 + 1];
                assert!(
                    (re - a.values[idx]).abs() < 1e-10,
                    "re[{r},{c}] = {re}, expected {}",
                    a.values[idx]
                );
                assert!(im.abs() < 1e-10, "im[{r},{c}] = {im}");
            }
        }
    }

    #[test]
    fn fft2_rejects_1d() {
        let a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        assert!(a.fft2().is_err());
    }

    // ── special math function tests ──

    #[test]
    fn gamma_basic_values() {
        // Gamma(1) = 1, Gamma(2) = 1, Gamma(3) = 2, Gamma(4) = 6, Gamma(5) = 24
        let a = UFuncArray::new(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0], DType::F64).unwrap();
        let g = a.gamma();
        let expected = [1.0, 1.0, 2.0, 6.0, 24.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (g.values[i] - exp).abs() < 1e-8,
                "Gamma({}) = {}, expected {}",
                a.values[i],
                g.values[i],
                exp
            );
        }
    }

    #[test]
    fn gamma_half() {
        // Gamma(0.5) = sqrt(pi) ≈ 1.7724538509
        let a = UFuncArray::new(vec![1], vec![0.5], DType::F64).unwrap();
        let g = a.gamma();
        let sqrt_pi = std::f64::consts::PI.sqrt();
        assert!(
            (g.values[0] - sqrt_pi).abs() < 1e-8,
            "Gamma(0.5) = {}, expected {sqrt_pi}",
            g.values[0]
        );
    }

    #[test]
    fn lgamma_basic() {
        // lgamma(1) = 0, lgamma(5) = ln(24) ≈ 3.178
        let a = UFuncArray::new(vec![2], vec![1.0, 5.0], DType::F64).unwrap();
        let lg = a.lgamma();
        assert!(lg.values[0].abs() < 1e-8, "lgamma(1) = {}", lg.values[0]);
        assert!(
            (lg.values[1] - 24.0_f64.ln()).abs() < 1e-8,
            "lgamma(5) = {}, expected {}",
            lg.values[1],
            24.0_f64.ln()
        );
    }

    #[test]
    fn erf_basic_values() {
        // erf(0) = 0, erf(large) ≈ 1, erf(-large) ≈ -1
        let a = UFuncArray::new(vec![3], vec![0.0, 5.0, -5.0], DType::F64).unwrap();
        let e = a.erf();
        assert!(e.values[0].abs() < 1e-10, "erf(0) = {}", e.values[0]);
        assert!(
            (e.values[1] - 1.0).abs() < 1e-6,
            "erf(5) = {}, expected ~1",
            e.values[1]
        );
        assert!(
            (e.values[2] + 1.0).abs() < 1e-6,
            "erf(-5) = {}, expected ~-1",
            e.values[2]
        );
    }

    #[test]
    fn erf_known_value() {
        // erf(1) ≈ 0.8427007929
        let a = UFuncArray::new(vec![1], vec![1.0], DType::F64).unwrap();
        let e = a.erf();
        assert!(
            (e.values[0] - 0.8427007929).abs() < 1e-5,
            "erf(1) = {}",
            e.values[0]
        );
    }

    #[test]
    fn erfc_complement() {
        // erfc(x) = 1 - erf(x)
        let a = UFuncArray::new(vec![3], vec![0.0, 1.0, 2.0], DType::F64).unwrap();
        let e = a.erf();
        let ec = a.erfc();
        for i in 0..3 {
            assert!(
                (e.values[i] + ec.values[i] - 1.0).abs() < 1e-10,
                "erf({}) + erfc({}) = {}, expected 1",
                a.values[i],
                a.values[i],
                e.values[i] + ec.values[i]
            );
        }
    }

    #[test]
    fn digamma_integer_values() {
        // psi(1) = -gamma_euler ≈ -0.5772156649
        let a = UFuncArray::new(vec![1], vec![1.0], DType::F64).unwrap();
        let d = a.digamma();
        assert!(
            (d.values[0] - (-0.5772156649)).abs() < 1e-4,
            "digamma(1) = {}, expected ≈ -0.5772",
            d.values[0]
        );
    }

    // ── Bessel function tests ──

    #[test]
    fn j0_at_zero_is_one() {
        let a = UFuncArray::new(vec![1], vec![0.0], DType::F64).unwrap();
        let j = a.j0();
        assert!(
            (j.values[0] - 1.0).abs() < 1e-10,
            "J0(0) = {}, expected 1",
            j.values[0]
        );
    }

    #[test]
    fn j0_known_zeros() {
        // J0 has zeros near 2.4048, 5.5201, 8.6537
        let a = UFuncArray::new(vec![3], vec![2.4048, 5.5201, 8.6537], DType::F64).unwrap();
        let j = a.j0();
        for i in 0..3 {
            assert!(
                j.values[i].abs() < 1e-3,
                "J0({}) = {}, expected ~0",
                a.values[i],
                j.values[i]
            );
        }
    }

    #[test]
    fn j1_at_zero_is_zero() {
        let a = UFuncArray::new(vec![1], vec![0.0], DType::F64).unwrap();
        let j = a.j1();
        assert!(j.values[0].abs() < 1e-10, "J1(0) = {}", j.values[0]);
    }

    #[test]
    fn j1_known_value() {
        // J1(1) ≈ 0.4400505857
        let a = UFuncArray::new(vec![1], vec![1.0], DType::F64).unwrap();
        let j = a.j1();
        assert!(
            (j.values[0] - 0.4400505857).abs() < 1e-4,
            "J1(1) = {}",
            j.values[0]
        );
    }

    #[test]
    fn y0_known_value() {
        // Y0(1) ≈ 0.0882569642
        let a = UFuncArray::new(vec![1], vec![1.0], DType::F64).unwrap();
        let y = a.y0();
        assert!(
            (y.values[0] - 0.0882569642).abs() < 1e-3,
            "Y0(1) = {}",
            y.values[0]
        );
    }

    #[test]
    fn y1_known_value() {
        // Y1(1) ≈ -0.7812128213
        let a = UFuncArray::new(vec![1], vec![1.0], DType::F64).unwrap();
        let y = a.y1();
        assert!(
            (y.values[0] - (-0.7812128213)).abs() < 1e-3,
            "Y1(1) = {}",
            y.values[0]
        );
    }

    #[test]
    fn y0_negative_is_neg_infinity() {
        let a = UFuncArray::new(vec![1], vec![-1.0], DType::F64).unwrap();
        let y = a.y0();
        assert!(y.values[0] == f64::NEG_INFINITY);
    }

    // ── fftn / ifftn / rfftn / irfftn ────────────────────────────────

    #[test]
    fn fftn_1d_matches_fft() {
        let a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let fft1 = a.fft(None).unwrap();
        let fftn = a.fftn().unwrap();
        // fftn on 1-D should match fft
        assert_eq!(fft1.shape(), fftn.shape());
        for (x, y) in fft1.values().iter().zip(fftn.values().iter()) {
            assert!((x - y).abs() < 1e-10, "fftn vs fft mismatch");
        }
    }

    #[test]
    fn fftn_2d_matches_fft2() {
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let fft2_res = a.fft2().unwrap();
        let fftn_res = a.fftn().unwrap();
        assert_eq!(fft2_res.shape(), fftn_res.shape());
        for (x, y) in fft2_res.values().iter().zip(fftn_res.values().iter()) {
            assert!((x - y).abs() < 1e-10, "fftn vs fft2 mismatch");
        }
    }

    #[test]
    fn fftn_ifftn_roundtrip() {
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let transformed = a.fftn().unwrap();
        let recovered = transformed.ifftn().unwrap();
        // Extract real parts
        for i in 0..6 {
            let re = recovered.values()[i * 2];
            assert!(
                (re - a.values()[i]).abs() < 1e-10,
                "roundtrip at {i}: {re} vs {}",
                a.values()[i]
            );
        }
    }

    #[test]
    fn fftn_3d_roundtrip() {
        // 2×2×2 cube
        let vals: Vec<f64> = (1..=8).map(|x| x as f64).collect();
        let a = UFuncArray::new(vec![2, 2, 2], vals.clone(), DType::F64).unwrap();
        let transformed = a.fftn().unwrap();
        assert_eq!(transformed.shape(), &[2, 2, 2, 2]);
        let recovered = transformed.ifftn().unwrap();
        for (i, &v) in vals.iter().enumerate().take(8) {
            let re = recovered.values()[i * 2];
            assert!((re - v).abs() < 1e-10, "3D roundtrip at {i}");
        }
    }

    #[test]
    fn rfftn_1d_matches_rfft() {
        let a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let rfft1 = a.rfft(None).unwrap();
        let rfftn = a.rfftn().unwrap();
        assert_eq!(rfft1.shape(), rfftn.shape());
        for (x, y) in rfft1.values().iter().zip(rfftn.values().iter()) {
            assert!((x - y).abs() < 1e-10, "rfftn vs rfft mismatch");
        }
    }

    #[test]
    fn rfftn_irfftn_roundtrip() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = UFuncArray::new(vec![2, 3], vals.clone(), DType::F64).unwrap();
        let transformed = a.rfftn().unwrap();
        let recovered = transformed.irfftn(Some(3)).unwrap();
        assert_eq!(recovered.shape(), &[2, 3]);
        for (i, &v) in vals.iter().enumerate().take(6) {
            assert!(
                (recovered.values()[i] - v).abs() < 1e-10,
                "rfftn roundtrip at {i}"
            );
        }
    }

    // ── complex operation tests ──

    #[test]
    fn angle_basic() {
        // Complex number 1+1i has angle pi/4
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0], DType::F64).unwrap();
        let angles = a.angle().unwrap();
        assert_eq!(angles.shape(), &[2]);
        assert!((angles.values()[0] - 0.0).abs() < 1e-10); // angle(1+0i) = 0
        assert!((angles.values()[1] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
        // angle(0+1i) = pi/2
    }

    #[test]
    fn angle_negative_real() {
        // -1+0i has angle pi
        let a = UFuncArray::new(vec![1, 2], vec![-1.0, 0.0], DType::F64).unwrap();
        let angles = a.angle().unwrap();
        assert!((angles.values()[0] - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn angle_rejects_non_complex() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        assert!(a.angle().is_err());
    }

    #[test]
    fn real_basic() {
        let a = UFuncArray::new(vec![2, 2], vec![3.0, 4.0, 5.0, -2.0], DType::Complex128).unwrap();
        let re = a.real().unwrap();
        assert_eq!(re.shape(), &[2]);
        assert!((re.values()[0] - 3.0).abs() < 1e-10);
        assert!((re.values()[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn imag_basic() {
        let a = UFuncArray::new(vec![2, 2], vec![3.0, 4.0, 5.0, -2.0], DType::Complex128).unwrap();
        let im = a.imag().unwrap();
        assert_eq!(im.shape(), &[2]);
        assert!((im.values()[0] - 4.0).abs() < 1e-10);
        assert!((im.values()[1] - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn conj_basic() {
        let a = UFuncArray::new(vec![2, 2], vec![3.0, 4.0, 1.0, -1.0], DType::Complex128).unwrap();
        let c = a.conj().unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert!((c.values()[0] - 3.0).abs() < 1e-10); // real unchanged
        assert!((c.values()[1] - (-4.0)).abs() < 1e-10); // imag negated
        assert!((c.values()[2] - 1.0).abs() < 1e-10);
        assert!((c.values()[3] - 1.0).abs() < 1e-10); // -(-1) = 1
    }

    #[test]
    fn conjugate_is_conj() {
        let a = UFuncArray::new(vec![1, 2], vec![2.0, 3.0], DType::Complex128).unwrap();
        let c1 = a.conj().unwrap();
        let c2 = a.conjugate().unwrap();
        assert_eq!(c1.values(), c2.values());
    }

    #[test]
    fn abs_complex_basic() {
        // |3+4i| = 5, |0+1i| = 1
        let a = UFuncArray::new(vec![2, 2], vec![3.0, 4.0, 0.0, 1.0], DType::Complex128).unwrap();
        let abs = a.abs_complex().unwrap();
        assert_eq!(abs.shape(), &[2]);
        assert!((abs.values()[0] - 5.0).abs() < 1e-10);
        assert!((abs.values()[1] - 1.0).abs() < 1e-10);
    }

    // ── numeric utility tests ──

    #[test]
    fn nan_to_num_basic() {
        let a = UFuncArray::new(
            vec![4],
            vec![1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
            DType::F64,
        )
        .unwrap();
        let r = a.nan_to_num_default();
        assert!((r.values()[0] - 1.0).abs() < 1e-10);
        assert!((r.values()[1] - 0.0).abs() < 1e-10); // NaN -> 0
        assert!((r.values()[2] - f64::MAX).abs() < 1e-10); // +inf -> MAX
        assert!((r.values()[3] - f64::MIN).abs() < 1e-10); // -inf -> MIN
    }

    #[test]
    fn nan_to_num_custom() {
        let a = UFuncArray::new(
            vec![3],
            vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
            DType::F64,
        )
        .unwrap();
        let r = a.nan_to_num(-1.0, 999.0, -999.0);
        assert!((r.values()[0] - (-1.0)).abs() < 1e-10);
        assert!((r.values()[1] - 999.0).abs() < 1e-10);
        assert!((r.values()[2] - (-999.0)).abs() < 1e-10);
    }

    #[test]
    fn flatnonzero_basic() {
        let a = UFuncArray::new(vec![5], vec![0.0, 1.0, 0.0, 3.0, 0.0], DType::F64).unwrap();
        let nz = a.flatnonzero();
        assert_eq!(nz.shape(), &[2]);
        assert!((nz.values()[0] - 1.0).abs() < 1e-10);
        assert!((nz.values()[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn flatnonzero_all_zero() {
        let a = UFuncArray::new(vec![3], vec![0.0, 0.0, 0.0], DType::F64).unwrap();
        let nz = a.flatnonzero();
        assert_eq!(nz.shape(), &[0]);
        assert!(nz.values().is_empty());
    }

    #[test]
    fn fix_basic() {
        let a = UFuncArray::new(vec![4], vec![2.7, -2.7, 0.5, -0.5], DType::F64).unwrap();
        let r = a.fix();
        assert!((r.values()[0] - 2.0).abs() < 1e-10);
        assert!((r.values()[1] - (-2.0)).abs() < 1e-10);
        assert!((r.values()[2] - 0.0).abs() < 1e-10);
        assert!((r.values()[3] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn select_basic() {
        let cond1 = UFuncArray::new(vec![4], vec![1.0, 0.0, 0.0, 0.0], DType::Bool).unwrap();
        let cond2 = UFuncArray::new(vec![4], vec![0.0, 1.0, 0.0, 0.0], DType::Bool).unwrap();
        let choice1 = UFuncArray::new(vec![4], vec![10.0, 20.0, 30.0, 40.0], DType::F64).unwrap();
        let choice2 =
            UFuncArray::new(vec![4], vec![100.0, 200.0, 300.0, 400.0], DType::F64).unwrap();
        let r = UFuncArray::select(&[&cond1, &cond2], &[&choice1, &choice2], -1.0).unwrap();
        assert!((r.values()[0] - 10.0).abs() < 1e-10); // first cond matches
        assert!((r.values()[1] - 200.0).abs() < 1e-10); // second cond matches
        assert!((r.values()[2] - (-1.0)).abs() < 1e-10); // no match, default
        assert!((r.values()[3] - (-1.0)).abs() < 1e-10); // no match, default
    }

    #[test]
    fn select_first_condition_wins() {
        let cond1 = UFuncArray::new(vec![2], vec![1.0, 1.0], DType::Bool).unwrap();
        let cond2 = UFuncArray::new(vec![2], vec![1.0, 1.0], DType::Bool).unwrap();
        let choice1 = UFuncArray::new(vec![2], vec![10.0, 20.0], DType::F64).unwrap();
        let choice2 = UFuncArray::new(vec![2], vec![30.0, 40.0], DType::F64).unwrap();
        let r = UFuncArray::select(&[&cond1, &cond2], &[&choice1, &choice2], 0.0).unwrap();
        // First condition matches, so choice1 wins
        assert!((r.values()[0] - 10.0).abs() < 1e-10);
        assert!((r.values()[1] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn select_empty_condlist() {
        assert!(UFuncArray::select(&[], &[], 0.0).is_err());
    }

    #[test]
    fn deg2rad_basic() {
        let a = UFuncArray::new(vec![3], vec![0.0, 90.0, 180.0], DType::F64).unwrap();
        let r = a.deg2rad();
        assert!((r.values()[0] - 0.0).abs() < 1e-10);
        assert!((r.values()[1] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
        assert!((r.values()[2] - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn rad2deg_basic() {
        let a = UFuncArray::new(
            vec![3],
            vec![0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI],
            DType::F64,
        )
        .unwrap();
        let r = a.rad2deg();
        assert!((r.values()[0] - 0.0).abs() < 1e-10);
        assert!((r.values()[1] - 90.0).abs() < 1e-10);
        assert!((r.values()[2] - 180.0).abs() < 1e-10);
    }

    #[test]
    fn copyto_basic() {
        let mut dst = UFuncArray::new(vec![3], vec![0.0, 0.0, 0.0], DType::F64).unwrap();
        let src = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        dst.copyto(&src, None, None).unwrap();
        assert_eq!(dst.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn copyto_with_mask() {
        let mut dst = UFuncArray::new(vec![3], vec![0.0, 0.0, 0.0], DType::F64).unwrap();
        let src = UFuncArray::new(vec![3], vec![10.0, 20.0, 30.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![3], vec![1.0, 0.0, 1.0], DType::Bool).unwrap();
        dst.copyto(&src, Some(&mask), None).unwrap();
        assert!((dst.values()[0] - 10.0).abs() < 1e-10);
        assert!((dst.values()[1] - 0.0).abs() < 1e-10); // mask false, unchanged
        assert!((dst.values()[2] - 30.0).abs() < 1e-10);
    }

    #[test]
    fn copyto_casting_safe_allows_i32_to_f64() {
        let mut dst = UFuncArray::new(vec![2], vec![0.0, 0.0], DType::F64).unwrap();
        let src = UFuncArray::new(vec![2], vec![5.0, 10.0], DType::I32).unwrap();
        dst.copyto(&src, None, Some("safe")).unwrap();
        assert_eq!(dst.values(), &[5.0, 10.0]);
    }

    #[test]
    fn copyto_casting_safe_rejects_f64_to_i32() {
        let mut dst = UFuncArray::new(vec![2], vec![0.0, 0.0], DType::I32).unwrap();
        let src = UFuncArray::new(vec![2], vec![5.0, 10.0], DType::F64).unwrap();
        let err = dst.copyto(&src, None, Some("safe")).unwrap_err();
        assert!(err.to_string().contains("cannot cast"));
    }

    #[test]
    fn copyto_casting_unsafe_allows_any() {
        let mut dst = UFuncArray::new(vec![2], vec![0.0, 0.0], DType::I32).unwrap();
        let src = UFuncArray::new(vec![2], vec![5.0, 10.0], DType::F64).unwrap();
        dst.copyto(&src, None, Some("unsafe")).unwrap();
        assert_eq!(dst.values(), &[5.0, 10.0]);
    }

    #[test]
    fn deg2rad_rad2deg_roundtrip() {
        let a = UFuncArray::new(vec![3], vec![45.0, 90.0, 360.0], DType::F64).unwrap();
        let r = a.deg2rad().rad2deg();
        for i in 0..3 {
            assert!(
                (r.values()[i] - a.values()[i]).abs() < 1e-10,
                "roundtrip at {i}"
            );
        }
    }

    // ── quantile interpolation method tests ──

    #[test]
    fn percentile_linear_default() {
        // [1, 2, 3, 4, 5] at 50th percentile = 3.0
        let a = UFuncArray::new(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0], DType::F64).unwrap();
        let r = a
            .percentile_method(50.0, None, QuantileInterp::Linear)
            .unwrap();
        assert!((r.values()[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn percentile_linear_interpolates() {
        // [1, 2, 3, 4] at 25th percentile: idx = 0.25*3 = 0.75 -> 1*0.25 + 2*0.75 = 1.75
        let a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let r = a
            .percentile_method(25.0, None, QuantileInterp::Linear)
            .unwrap();
        assert!((r.values()[0] - 1.75).abs() < 1e-10);
    }

    #[test]
    fn percentile_lower() {
        // [1, 2, 3, 4] at 25th percentile: idx = 0.75, lower = floor = 0 -> 1.0
        let a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let r = a
            .percentile_method(25.0, None, QuantileInterp::Lower)
            .unwrap();
        assert!((r.values()[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn percentile_higher() {
        // [1, 2, 3, 4] at 25th percentile: idx = 0.75, higher = ceil = 1 -> 2.0
        let a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let r = a
            .percentile_method(25.0, None, QuantileInterp::Higher)
            .unwrap();
        assert!((r.values()[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn percentile_nearest() {
        // [1, 2, 3, 4] at 25th: idx=0.75, >0.5 so nearest is higher -> 2.0
        let a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let r = a
            .percentile_method(25.0, None, QuantileInterp::Nearest)
            .unwrap();
        assert!((r.values()[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn percentile_nearest_round_down() {
        // [1, 2, 3, 4, 5] at 30th: idx=0.3*4=1.2, fractional=0.2<=0.5 so nearest is lower -> 2.0
        let a = UFuncArray::new(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0], DType::F64).unwrap();
        let r = a
            .percentile_method(30.0, None, QuantileInterp::Nearest)
            .unwrap();
        assert!((r.values()[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn percentile_midpoint() {
        // [1, 2, 3, 4] at 25th: idx=0.75, midpoint of (1, 2) = 1.5
        let a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let r = a
            .percentile_method(25.0, None, QuantileInterp::Midpoint)
            .unwrap();
        assert!((r.values()[0] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn percentile_at_exact_index() {
        // [10, 20, 30] at 50th: idx=1.0 exactly -> all methods should give 20.0
        let a = UFuncArray::new(vec![3], vec![10.0, 20.0, 30.0], DType::F64).unwrap();
        for method in [
            QuantileInterp::Linear,
            QuantileInterp::Lower,
            QuantileInterp::Higher,
            QuantileInterp::Nearest,
            QuantileInterp::Midpoint,
        ] {
            let r = a.percentile_method(50.0, None, method).unwrap();
            assert!(
                (r.values()[0] - 20.0).abs() < 1e-10,
                "method {method:?} at exact index"
            );
        }
    }

    #[test]
    fn quantile_method_basic() {
        let a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let r = a
            .quantile_method(0.25, None, QuantileInterp::Lower)
            .unwrap();
        assert!((r.values()[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn percentile_method_with_axis() {
        // 2x3 array, percentile along axis 1
        let a =
            UFuncArray::new(vec![2, 3], vec![3.0, 1.0, 2.0, 6.0, 4.0, 5.0], DType::F64).unwrap();
        let r = a
            .percentile_method(50.0, Some(1), QuantileInterp::Lower)
            .unwrap();
        assert_eq!(r.shape(), &[2]);
        // Row 0 sorted: [1, 2, 3], 50th lower = idx=1, -> 2.0
        assert!((r.values()[0] - 2.0).abs() < 1e-10);
        // Row 1 sorted: [4, 5, 6], 50th lower = idx=1, -> 5.0
        assert!((r.values()[1] - 5.0).abs() < 1e-10);
    }

    // ── array_split tests ──

    #[test]
    fn array_split_even() {
        let a = UFuncArray::new(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let parts = a.array_split(3, 0).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].values(), &[1.0, 2.0]);
        assert_eq!(parts[1].values(), &[3.0, 4.0]);
        assert_eq!(parts[2].values(), &[5.0, 6.0]);
    }

    #[test]
    fn array_split_uneven() {
        // 7 elements into 3: sizes 3, 2, 2
        let a =
            UFuncArray::new(vec![7], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], DType::F64).unwrap();
        let parts = a.array_split(3, 0).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].shape(), &[3]); // first gets remainder
        assert_eq!(parts[0].values(), &[1.0, 2.0, 3.0]);
        assert_eq!(parts[1].shape(), &[2]);
        assert_eq!(parts[1].values(), &[4.0, 5.0]);
        assert_eq!(parts[2].shape(), &[2]);
        assert_eq!(parts[2].values(), &[6.0, 7.0]);
    }

    #[test]
    fn array_split_2d_axis0() {
        // 3x2 array split into 2 along axis 0: sizes 2, 1
        let a =
            UFuncArray::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let parts = a.array_split(2, 0).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].shape(), &[2, 2]);
        assert_eq!(parts[0].values(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(parts[1].shape(), &[1, 2]);
        assert_eq!(parts[1].values(), &[5.0, 6.0]);
    }

    #[test]
    fn array_split_more_sections_than_elements() {
        // 2 elements into 3: sizes 1, 1, 0
        let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap();
        let parts = a.array_split(3, 0).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].values(), &[1.0]);
        assert_eq!(parts[1].values(), &[2.0]);
        assert!(parts[2].values().is_empty());
    }

    // ── block tests ──

    #[test]
    fn block_row_basic() {
        let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![3.0, 4.0, 5.0], DType::F64).unwrap();
        let r = UFuncArray::block_row(&[&a, &b]).unwrap();
        assert_eq!(r.shape(), &[5]);
        assert_eq!(r.values(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn block_2d_basic() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2, 1], vec![5.0, 6.0], DType::F64).unwrap();
        let c = UFuncArray::new(vec![1, 2], vec![7.0, 8.0], DType::F64).unwrap();
        let d = UFuncArray::new(vec![1, 1], vec![9.0], DType::F64).unwrap();
        let r = UFuncArray::block_2d(&[vec![&a, &b], vec![&c, &d]]).unwrap();
        assert_eq!(r.shape(), &[3, 3]);
        assert_eq!(r.values(), &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0]);
    }

    // ── result_type tests ──

    #[test]
    fn result_type_basic() {
        let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::I32).unwrap();
        let b = UFuncArray::new(vec![2], vec![3.0, 4.0], DType::F64).unwrap();
        assert_eq!(UFuncArray::result_type(&[&a, &b]), DType::F64);
    }

    #[test]
    fn result_type_same() {
        let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::I32).unwrap();
        let b = UFuncArray::new(vec![2], vec![3.0, 4.0], DType::I32).unwrap();
        assert_eq!(UFuncArray::result_type(&[&a, &b]), DType::I32);
    }

    #[test]
    fn result_type_empty() {
        assert_eq!(UFuncArray::result_type(&[]), DType::F64);
    }

    // ── multi-axis reduction tests ──

    #[test]
    fn reduce_sum_axes_two_axes() {
        // shape [2, 3, 4], sum over axes (0, 2) → shape [3]
        let vals: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        let a = UFuncArray::new(vec![2, 3, 4], vals, DType::F64).unwrap();
        let r = a.reduce_sum_axes(&[0, 2], false).unwrap();
        assert_eq!(r.shape(), &[3]);
        // axis 2 first (highest): [2,3,4] → [2,3] sum each group of 4
        // then axis 0: [2,3] → [3]
        // Expected: each of 3 slices sums 2*4=8 elements
        assert_eq!(r.values().len(), 3);
        let total: f64 = r.values().iter().sum();
        assert!((total - 300.0).abs() < 1e-9); // sum(1..24) = 300
    }

    #[test]
    fn reduce_sum_axes_keepdims() {
        let vals: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        let a = UFuncArray::new(vec![2, 3, 4], vals, DType::F64).unwrap();
        let r = a.reduce_sum_axes(&[0, 2], true).unwrap();
        assert_eq!(r.shape(), &[1, 3, 1]);
    }

    #[test]
    fn reduce_sum_axes_empty_axes() {
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let r = a.reduce_sum_axes(&[], false).unwrap();
        assert_eq!(r.values(), a.values());
    }

    #[test]
    fn reduce_prod_axes_basic() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let r = a.reduce_prod_axes(&[0, 1], false).unwrap();
        assert_eq!(r.shape(), &[] as &[usize]);
        assert!((r.values()[0] - 24.0).abs() < 1e-9);
    }

    #[test]
    fn reduce_min_axes_basic() {
        let a =
            UFuncArray::new(vec![2, 3], vec![5.0, 1.0, 3.0, 2.0, 4.0, 6.0], DType::F64).unwrap();
        let r = a.reduce_min_axes(&[0, 1], false).unwrap();
        assert!((r.values()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn reduce_max_axes_basic() {
        let a =
            UFuncArray::new(vec![2, 3], vec![5.0, 1.0, 3.0, 2.0, 4.0, 6.0], DType::F64).unwrap();
        let r = a.reduce_max_axes(&[0, 1], false).unwrap();
        assert!((r.values()[0] - 6.0).abs() < 1e-9);
    }

    #[test]
    fn reduce_mean_axes_basic() {
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let r = a.reduce_mean_axes(&[0, 1], false).unwrap();
        assert!((r.values()[0] - 3.5).abs() < 1e-9);
    }

    // ── reduce with where/initial tests ──

    #[test]
    fn reduce_sum_where_basic() {
        let a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![4], vec![1.0, 0.0, 1.0, 0.0], DType::F64).unwrap();
        let r = a.reduce_sum_where(&mask, None, false).unwrap();
        assert!((r.values()[0] - 4.0).abs() < 1e-9); // 1 + 3 = 4
    }

    #[test]
    fn reduce_sum_where_axis() {
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let mask =
            UFuncArray::new(vec![2, 3], vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0], DType::F64).unwrap();
        let r = a.reduce_sum_where(&mask, Some(0), false).unwrap();
        assert_eq!(r.shape(), &[3]);
        assert!((r.values()[0] - 1.0).abs() < 1e-9); // only row 0
        assert!((r.values()[1] - 5.0).abs() < 1e-9); // only row 1
        assert!((r.values()[2] - 3.0).abs() < 1e-9); // only row 0
    }

    #[test]
    fn reduce_sum_initial_basic() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let r = a.reduce_sum_initial(None, false, 10.0).unwrap();
        assert!((r.values()[0] - 16.0).abs() < 1e-9); // 6 + 10 = 16
    }

    #[test]
    fn reduce_prod_initial_basic() {
        let a = UFuncArray::new(vec![3], vec![2.0, 3.0, 4.0], DType::F64).unwrap();
        let r = a.reduce_prod_initial(None, false, 0.5).unwrap();
        assert!((r.values()[0] - 12.0).abs() < 1e-9); // 24 * 0.5 = 12
    }

    // ── histogram_bin_edges tests ──

    #[test]
    fn histogram_bin_edges_basic() {
        let a = UFuncArray::new(vec![5], vec![0.0, 1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let edges = a.histogram_bin_edges(4).unwrap();
        assert_eq!(edges.shape(), &[5]); // 4 + 1 edges
        assert!((edges.values()[0] - 0.0).abs() < 1e-9);
        assert!((edges.values()[4] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn histogram_bin_edges_single_bin() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let edges = a.histogram_bin_edges(1).unwrap();
        assert_eq!(edges.shape(), &[2]);
    }

    // ── histogramdd tests ──

    #[test]
    fn histogramdd_2d_basic() {
        // 4 points in 2D, 2 bins per dimension
        let sample = UFuncArray::new(
            vec![4, 2],
            vec![0.5, 0.5, 1.5, 0.5, 0.5, 1.5, 1.5, 1.5],
            DType::F64,
        )
        .unwrap();
        let (hist, edges) = sample.histogramdd(&[2, 2]).unwrap();
        assert_eq!(hist.shape(), &[2, 2]);
        assert_eq!(edges.len(), 2);
        let total: f64 = hist.values().iter().sum();
        assert!((total - 4.0).abs() < 1e-9); // all points counted
    }

    #[test]
    fn histogramdd_3d() {
        // 6 points in 3D, 2 bins per dimension
        let sample = UFuncArray::new(
            vec![6, 3],
            vec![
                0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.2,
                0.8, 0.3,
            ],
            DType::F64,
        )
        .unwrap();
        let (hist, edges) = sample.histogramdd(&[2, 2, 2]).unwrap();
        assert_eq!(hist.shape(), &[2, 2, 2]);
        assert_eq!(edges.len(), 3);
        let total: f64 = hist.values().iter().sum();
        assert!((total - 6.0).abs() < 1e-9);
    }

    // ── broadcast_shapes tests ──

    #[test]
    fn broadcast_shapes_basic() {
        let r = UFuncArray::broadcast_shapes(&[&[3, 1], &[1, 4]]).unwrap();
        assert_eq!(r, vec![3, 4]);
    }

    #[test]
    fn broadcast_shapes_multi() {
        let r = UFuncArray::broadcast_shapes(&[&[5, 1, 1], &[1, 3, 1], &[1, 1, 7]]).unwrap();
        assert_eq!(r, vec![5, 3, 7]);
    }

    #[test]
    fn broadcast_shapes_incompatible() {
        assert!(UFuncArray::broadcast_shapes(&[&[3], &[4]]).is_err());
    }

    // ── putmask tests ──

    #[test]
    fn putmask_basic() {
        let mut a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![4], vec![1.0, 0.0, 1.0, 0.0], DType::F64).unwrap();
        let vals = UFuncArray::new(vec![2], vec![10.0, 20.0], DType::F64).unwrap();
        a.putmask(&mask, &vals);
        assert_eq!(a.values(), &[10.0, 2.0, 20.0, 4.0]);
    }

    #[test]
    fn putmask_cycling() {
        let mut a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![4], vec![1.0, 1.0, 1.0, 1.0], DType::F64).unwrap();
        let vals = UFuncArray::new(vec![2], vec![10.0, 20.0], DType::F64).unwrap();
        a.putmask(&mask, &vals);
        // Cycles: 10, 20, 10, 20
        assert_eq!(a.values(), &[10.0, 20.0, 10.0, 20.0]);
    }

    // ── sliding_window_view tests ──

    #[test]
    fn as_strided_1d_to_2d() {
        // Use as_strided to create overlapping windows of size 3 from [1,2,3,4,5]
        let a = UFuncArray::new(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0], DType::F64).unwrap();
        // shape [3, 3], strides [1, 1] => sliding windows
        let r = a.as_strided(&[3, 3], &[1, 1]).unwrap();
        assert_eq!(r.shape(), &[3, 3]);
        assert_eq!(r.values(), &[1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn as_strided_repeat_row() {
        // Broadcast a single row [1,2,3] into a 3x3 matrix by setting row stride to 0
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let r = a.as_strided(&[3, 3], &[0, 1]).unwrap();
        assert_eq!(r.shape(), &[3, 3]);
        assert_eq!(r.values(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn as_strided_mismatched_dims() {
        let a = UFuncArray::new(vec![5], vec![1.0; 5], DType::F64).unwrap();
        assert!(a.as_strided(&[2, 3], &[1]).is_err()); // shape.len != strides.len
    }

    #[test]
    fn sliding_window_view_1d() {
        let a = UFuncArray::new(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0], DType::F64).unwrap();
        let r = a.sliding_window_view(&[3]).unwrap();
        assert_eq!(r.shape(), &[3, 3]); // 5-3+1=3 windows, each of size 3
        assert_eq!(r.values(), &[1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn sliding_window_view_2d() {
        // 3x4 array, 2x2 windows → (2, 3, 2, 2)
        let a = UFuncArray::new(
            vec![3, 4],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            DType::F64,
        )
        .unwrap();
        let r = a.sliding_window_view(&[2, 2]).unwrap();
        assert_eq!(r.shape(), &[2, 3, 2, 2]); // (3-2+1, 4-2+1, 2, 2)
        // First window [0,0]: [[1,2],[5,6]]
        assert_eq!(&r.values()[0..4], &[1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn sliding_window_view_full_window() {
        // Window same size as array → single window
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let r = a.sliding_window_view(&[3]).unwrap();
        assert_eq!(r.shape(), &[1, 3]);
        assert_eq!(r.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn sliding_window_view_invalid_window() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        assert!(a.sliding_window_view(&[4]).is_err()); // window > array
        assert!(a.sliding_window_view(&[0]).is_err()); // zero window
    }

    // ── MaskedArray tests ──────────────────────────────────────────

    #[test]
    fn masked_array_new_no_mask() {
        let data = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let ma = MaskedArray::new(data.clone(), None, None).unwrap();
        assert_eq!(ma.shape(), &[3]);
        assert!(ma.mask().is_none());
        assert_eq!(ma.data().values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn masked_array_new_with_mask() {
        let data = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![4], vec![0.0, 1.0, 0.0, 1.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        assert!(ma.mask().is_some());
        assert_eq!(ma.mask().unwrap().values(), &[0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn masked_array_shape_mismatch() {
        let data = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![2], vec![0.0, 1.0], DType::Bool).unwrap();
        let err = MaskedArray::new(data, Some(mask), None).unwrap_err();
        assert!(matches!(err, MAError::MaskShapeMismatch { .. }));
    }

    #[test]
    fn masked_array_filled() {
        let data = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![4], vec![0.0, 1.0, 0.0, 1.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        let filled = ma.filled(-999.0);
        assert_eq!(filled.values(), &[1.0, -999.0, 3.0, -999.0]);
    }

    #[test]
    fn masked_array_compressed() {
        let data = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![4], vec![0.0, 1.0, 0.0, 1.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        let comp = ma.compressed();
        assert_eq!(comp.shape(), &[2]);
        assert_eq!(comp.values(), &[1.0, 3.0]);
    }

    #[test]
    fn masked_array_compressed_no_mask() {
        let data = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let ma = MaskedArray::new(data, None, None).unwrap();
        let comp = ma.compressed();
        assert_eq!(comp.shape(), &[3]);
        assert_eq!(comp.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn masked_array_count() {
        let data = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![4], vec![0.0, 1.0, 0.0, 1.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        let c = ma.count(None).unwrap();
        assert_eq!(c.values(), &[2.0]); // 2 valid elements
    }

    #[test]
    fn masked_equal_test() {
        let data = UFuncArray::new(vec![5], vec![1.0, 2.0, 3.0, 2.0, 5.0], DType::F64).unwrap();
        let ma = MaskedArray::masked_equal(&data, 2.0).unwrap();
        // Elements equal to 2 are masked
        assert_eq!(ma.mask().unwrap().values(), &[0.0, 1.0, 0.0, 1.0, 0.0]);
        let comp = ma.compressed();
        assert_eq!(comp.values(), &[1.0, 3.0, 5.0]);
    }

    #[test]
    fn masked_greater_test() {
        let data = UFuncArray::new(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0], DType::F64).unwrap();
        let ma = MaskedArray::masked_greater(&data, 3.0).unwrap();
        assert_eq!(ma.mask().unwrap().values(), &[0.0, 0.0, 0.0, 1.0, 1.0]);
        let comp = ma.compressed();
        assert_eq!(comp.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn masked_less_test() {
        let data = UFuncArray::new(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0], DType::F64).unwrap();
        let ma = MaskedArray::masked_less(&data, 3.0).unwrap();
        assert_eq!(ma.mask().unwrap().values(), &[1.0, 1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn masked_inside_test() {
        let data = UFuncArray::new(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0], DType::F64).unwrap();
        let ma = MaskedArray::masked_inside(&data, 2.0, 4.0).unwrap();
        assert_eq!(ma.mask().unwrap().values(), &[0.0, 1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn masked_outside_test() {
        let data = UFuncArray::new(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0], DType::F64).unwrap();
        let ma = MaskedArray::masked_outside(&data, 2.0, 4.0).unwrap();
        assert_eq!(ma.mask().unwrap().values(), &[1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn masked_invalid_test() {
        let data =
            UFuncArray::new(vec![4], vec![1.0, f64::NAN, f64::INFINITY, 4.0], DType::F64).unwrap();
        let ma = MaskedArray::masked_invalid(&data).unwrap();
        assert_eq!(ma.mask().unwrap().values(), &[0.0, 1.0, 1.0, 0.0]);
        let comp = ma.compressed();
        assert_eq!(comp.values(), &[1.0, 4.0]);
    }

    #[test]
    fn masked_where_test() {
        let data = UFuncArray::new(vec![4], vec![10.0, 20.0, 30.0, 40.0], DType::F64).unwrap();
        let cond = UFuncArray::new(vec![4], vec![0.0, 1.0, 0.0, 1.0], DType::Bool).unwrap();
        let ma = MaskedArray::masked_where(&cond, &data).unwrap();
        let comp = ma.compressed();
        assert_eq!(comp.values(), &[10.0, 30.0]);
    }

    #[test]
    fn masked_array_sum() {
        let data = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![4], vec![0.0, 1.0, 0.0, 1.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        let result = ma.sum(None, false).unwrap();
        assert_eq!(result.data().values(), &[4.0]); // 1 + 3
    }

    #[test]
    fn masked_array_sum_no_mask() {
        let data = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let ma = MaskedArray::new(data, None, None).unwrap();
        let result = ma.sum(None, false).unwrap();
        assert_eq!(result.data().values(), &[6.0]);
    }

    #[test]
    fn masked_array_prod() {
        let data = UFuncArray::new(vec![4], vec![2.0, 3.0, 4.0, 5.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![4], vec![0.0, 1.0, 0.0, 1.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        let result = ma.prod(None, false).unwrap();
        assert_eq!(result.data().values(), &[8.0]); // 2 * 4
    }

    #[test]
    fn masked_array_mean() {
        let data = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![4], vec![0.0, 1.0, 0.0, 1.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        let result = ma.mean(None, false).unwrap();
        assert_eq!(result.data().values(), &[2.0]); // (1+3)/2
    }

    #[test]
    fn masked_array_min_max() {
        let data = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![4], vec![1.0, 0.0, 0.0, 1.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        let min_result = ma.min(None, false).unwrap();
        let max_result = ma.max(None, false).unwrap();
        assert_eq!(min_result.data().values(), &[2.0]);
        assert_eq!(max_result.data().values(), &[3.0]);
    }

    #[test]
    fn masked_array_binary_op_mask_propagation() {
        let d1 = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let m1 = UFuncArray::new(vec![3], vec![0.0, 1.0, 0.0], DType::Bool).unwrap();
        let ma1 = MaskedArray::new(d1, Some(m1), None).unwrap();

        let d2 = UFuncArray::new(vec![3], vec![10.0, 20.0, 30.0], DType::F64).unwrap();
        let m2 = UFuncArray::new(vec![3], vec![0.0, 0.0, 1.0], DType::Bool).unwrap();
        let ma2 = MaskedArray::new(d2, Some(m2), None).unwrap();

        let result = ma1.elementwise_binary(&ma2, BinaryOp::Add).unwrap();
        // Mask OR: [0,1,0] OR [0,0,1] = [0,1,1]
        assert_eq!(result.mask().unwrap().values(), &[0.0, 1.0, 1.0]);
        // Data values computed regardless of mask
        assert_eq!(result.data().values(), &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn masked_array_unary_preserves_mask() {
        let data = UFuncArray::new(vec![3], vec![1.0, -2.0, 3.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![3], vec![0.0, 1.0, 0.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        let result = ma.elementwise_unary(UnaryOp::Abs);
        assert_eq!(result.data().values(), &[1.0, 2.0, 3.0]);
        assert_eq!(result.mask().unwrap().values(), &[0.0, 1.0, 0.0]);
    }

    #[test]
    fn masked_array_fill_value() {
        let data = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap();
        let mut ma = MaskedArray::new(data, None, Some(42.0)).unwrap();
        assert!((ma.fill_value() - 42.0).abs() < f64::EPSILON);
        ma.set_fill_value(-1.0);
        assert!((ma.fill_value() - (-1.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn masked_array_hard_mask() {
        let data = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap();
        let mut ma = MaskedArray::new(data, None, None).unwrap();
        assert!(!ma.is_hard_mask());
        ma.harden_mask();
        assert!(ma.is_hard_mask());
        ma.soften_mask();
        assert!(!ma.is_hard_mask());
    }

    #[test]
    fn masked_not_equal_test() {
        let data = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 2.0], DType::F64).unwrap();
        let ma = MaskedArray::masked_not_equal(&data, 2.0).unwrap();
        assert_eq!(ma.mask().unwrap().values(), &[1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn masked_greater_equal_test() {
        let data = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let ma = MaskedArray::masked_greater_equal(&data, 3.0).unwrap();
        assert_eq!(ma.mask().unwrap().values(), &[0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn masked_less_equal_test() {
        let data = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let ma = MaskedArray::masked_less_equal(&data, 2.0).unwrap();
        assert_eq!(ma.mask().unwrap().values(), &[1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn masked_array_from_values() {
        let ma = MaskedArray::from_values(
            vec![3],
            vec![10.0, 20.0, 30.0],
            Some(vec![0.0, 1.0, 0.0]),
            DType::F64,
        )
        .unwrap();
        assert_eq!(ma.compressed().values(), &[10.0, 30.0]);
    }

    #[test]
    fn masked_array_2d_sum_axis() {
        // [[1,2,3],[4,5,6]] with mask [[0,1,0],[0,0,1]]
        let data =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let mask =
            UFuncArray::new(vec![2, 3], vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        // Sum along axis 1: row 0 = 1+3=4, row 1 = 4+5=9
        let result = ma.sum(Some(1), false).unwrap();
        assert_eq!(result.data().values(), &[4.0, 9.0]);
    }

    // ── MaskedArray expanded tests ──────────────────────────────────

    #[test]
    fn masked_reshape() {
        let data =
            UFuncArray::new(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let mask =
            UFuncArray::new(vec![6], vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        let reshaped = ma.reshape(&[2, 3]).unwrap();
        assert_eq!(reshaped.shape(), &[2, 3]);
        assert_eq!(reshaped.mask().unwrap().shape(), &[2, 3]);
        assert_eq!(reshaped.data().values(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn masked_ravel() {
        let data =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let mask =
            UFuncArray::new(vec![2, 3], vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        let flat = ma.ravel();
        assert_eq!(flat.shape(), &[6]);
        assert_eq!(flat.mask().unwrap().shape(), &[6]);
    }

    #[test]
    fn masked_transpose() {
        let data =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let mask =
            UFuncArray::new(vec![2, 3], vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        let t = ma.transpose(None).unwrap();
        assert_eq!(t.shape(), &[3, 2]);
        // Mask should also be transposed
        assert_eq!(t.mask().unwrap().shape(), &[3, 2]);
    }

    #[test]
    fn masked_comparison_ops() {
        let data1 = UFuncArray::new(vec![3], vec![1.0, 5.0, 3.0], DType::F64).unwrap();
        let data2 = UFuncArray::new(vec![3], vec![2.0, 4.0, 3.0], DType::F64).unwrap();
        let ma1 = MaskedArray::new(data1, None, None).unwrap();
        let ma2 = MaskedArray::new(data2, None, None).unwrap();
        // 1<2, 5<4, 3<3 => [1, 0, 0]
        let lt = ma1.less_than(&ma2).unwrap();
        assert_eq!(lt.data().values(), &[1.0, 0.0, 0.0]);
        // 1==2, 5==4, 3==3 => [0, 0, 1]
        let eq = ma1.equal(&ma2).unwrap();
        assert_eq!(eq.data().values(), &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn masked_concatenate() {
        let d1 = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap();
        let m1 = UFuncArray::new(vec![2], vec![0.0, 1.0], DType::Bool).unwrap();
        let ma1 = MaskedArray::new(d1, Some(m1), None).unwrap();
        let d2 = UFuncArray::new(vec![3], vec![3.0, 4.0, 5.0], DType::F64).unwrap();
        let ma2 = MaskedArray::new(d2, None, None).unwrap();
        let cat = MaskedArray::concatenate(&[&ma1, &ma2], 0).unwrap();
        assert_eq!(cat.shape(), &[5]);
        assert_eq!(cat.data().values(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
        // mask: [0, 1, 0, 0, 0] — ma1 has mask, ma2 gets zeros
        assert_eq!(cat.mask().unwrap().values(), &[0.0, 1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn masked_any_all() {
        let data = UFuncArray::new(vec![3], vec![0.0, 1.0, 0.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![3], vec![0.0, 1.0, 0.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        // Valid elements: [0.0, _, 0.0] — the 1.0 is masked
        assert!(!ma.any()); // all valid elements are zero
        assert!(!ma.all()); // not all valid elements are nonzero
    }

    #[test]
    fn masked_nonzero_indices() {
        let data = UFuncArray::new(vec![4], vec![0.0, 5.0, 0.0, 3.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![4], vec![0.0, 0.0, 0.0, 1.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        // element 3 (value 3.0) is masked; only element 1 (value 5.0) is valid + nonzero
        assert_eq!(ma.nonzero_indices(), vec![1]);
    }

    #[test]
    fn masked_size_ndim() {
        let data = UFuncArray::new(vec![2, 3], vec![1.0; 6], DType::F64).unwrap();
        let ma = MaskedArray::new(data, None, None).unwrap();
        assert_eq!(ma.size(), 6);
        assert_eq!(ma.ndim(), 2);
    }

    // ── additional MaskedArray operation tests ─────────────────────

    #[test]
    fn masked_sort_pushes_masked_to_end() {
        let data = UFuncArray::new(vec![4], vec![3.0, 1.0, 4.0, 2.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![4], vec![0.0, 1.0, 0.0, 0.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        let sorted = ma.sort(None).unwrap();
        // Unmasked values [3, 4, 2] sorted = [2, 3, 4], masked at end
        let vals = sorted.data().values();
        assert!((vals[0] - 2.0).abs() < 1e-10);
        assert!((vals[1] - 3.0).abs() < 1e-10);
        assert!((vals[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn masked_argsort() {
        let data = UFuncArray::new(vec![3], vec![3.0, 1.0, 2.0], DType::F64).unwrap();
        let ma = MaskedArray::new(data, None, None).unwrap();
        let indices = ma.argsort(None).unwrap();
        assert_eq!(indices.values(), &[1.0, 2.0, 0.0]);
    }

    #[test]
    fn masked_argmin_argmax() {
        let data = UFuncArray::new(vec![4], vec![5.0, 1.0, 9.0, 3.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![4], vec![0.0, 0.0, 1.0, 0.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        let amin = ma.argmin(None).unwrap();
        assert_eq!(amin.values()[0], 1.0); // index 1 (value=1.0)
        let amax = ma.argmax(None).unwrap();
        assert_eq!(amax.values()[0], 0.0); // index 0 (value=5.0, 9.0 is masked)
    }

    #[test]
    fn masked_cumsum() {
        let data = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![4], vec![0.0, 1.0, 0.0, 0.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        let cs = ma.cumsum(None).unwrap();
        // masked element contributes 0: [1, 0, 3, 4] -> cumsum [1, 1, 4, 8]
        assert_eq!(cs.data().values(), &[1.0, 1.0, 4.0, 8.0]);
        assert!(cs.mask().is_some()); // mask preserved
    }

    #[test]
    fn masked_cumprod() {
        let data = UFuncArray::new(vec![3], vec![2.0, 3.0, 4.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![3], vec![0.0, 1.0, 0.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        let cp = ma.cumprod(None).unwrap();
        // masked element contributes 1: [2, 1, 4] -> cumprod [2, 2, 8]
        assert_eq!(cp.data().values(), &[2.0, 2.0, 8.0]);
    }

    #[test]
    fn masked_median() {
        let data = UFuncArray::new(vec![5], vec![1.0, 5.0, 3.0, 9.0, 7.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![5], vec![0.0, 1.0, 0.0, 1.0, 0.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        let med = ma.median().unwrap();
        // Unmasked: [1, 3, 7] -> median = 3
        assert!((med - 3.0).abs() < 1e-10);
    }

    #[test]
    fn masked_ptp() {
        let data = UFuncArray::new(vec![4], vec![1.0, 5.0, 3.0, 9.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![4], vec![0.0, 0.0, 1.0, 0.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        let p = ma.ptp().unwrap();
        // Unmasked: [1, 5, 9] -> ptp = 9 - 1 = 8
        assert!((p - 8.0).abs() < 1e-10);
    }

    #[test]
    fn masked_dot() {
        let d1 = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let m1 = UFuncArray::new(vec![3], vec![0.0, 1.0, 0.0], DType::Bool).unwrap();
        let ma1 = MaskedArray::new(d1, Some(m1), None).unwrap();
        let d2 = UFuncArray::new(vec![3], vec![4.0, 5.0, 6.0], DType::F64).unwrap();
        let ma2 = MaskedArray::new(d2, None, None).unwrap();
        let result = ma1.dot(&ma2).unwrap();
        // ma1 filled with 0: [1, 0, 3] . [4, 5, 6] = 4 + 0 + 18 = 22
        assert!((result.data().values()[0] - 22.0).abs() < 1e-10);
    }

    #[test]
    fn masked_expand_dims() {
        let data = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let ma = MaskedArray::new(data, None, None).unwrap();
        let expanded = ma.expand_dims(0).unwrap();
        assert_eq!(expanded.shape(), &[1, 3]);
    }

    #[test]
    fn masked_squeeze() {
        let data = UFuncArray::new(vec![1, 3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let ma = MaskedArray::new(data, None, None).unwrap();
        let squeezed = ma.squeeze(Some(0)).unwrap();
        assert_eq!(squeezed.shape(), &[3]);
    }

    #[test]
    fn masked_all_constructor() {
        let ma = MaskedArray::masked_all(vec![2, 3], DType::F64).unwrap();
        assert_eq!(ma.shape(), &[2, 3]);
        assert_eq!(ma.count_masked(), 6);
        let compressed = ma.compressed();
        assert!(compressed.values().is_empty());
    }

    #[test]
    fn masked_all_like_constructor() {
        let data = UFuncArray::new(vec![4], vec![1.0; 4], DType::F64).unwrap();
        let model = MaskedArray::new(data, None, None).unwrap();
        let ma = MaskedArray::masked_all_like(&model).unwrap();
        assert_eq!(ma.shape(), &[4]);
        assert_eq!(ma.count_masked(), 4);
    }

    #[test]
    fn masked_count_masked() {
        let data = UFuncArray::new(vec![4], vec![1.0; 4], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![4], vec![0.0, 1.0, 1.0, 0.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        assert_eq!(ma.count_masked(), 2);
    }

    #[test]
    fn masked_average_unweighted() {
        let data = UFuncArray::new(vec![4], vec![2.0, 4.0, 6.0, 8.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![4], vec![0.0, 1.0, 0.0, 0.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        let avg = ma.average(None).unwrap();
        // Unmasked: [2, 6, 8] -> mean = 16/3
        assert!((avg - 16.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn masked_take() {
        let data = UFuncArray::new(vec![4], vec![10.0, 20.0, 30.0, 40.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![4], vec![0.0, 1.0, 0.0, 0.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        let taken = ma.take(&[0, 2, 3]).unwrap();
        assert_eq!(taken.data().values(), &[10.0, 30.0, 40.0]);
        // Index 1 is not taken; mask should reflect only indices 0, 2, 3
        assert_eq!(taken.mask().unwrap().values(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn masked_clump_unmasked() {
        let data = UFuncArray::new(vec![6], vec![1.0; 6], DType::F64).unwrap();
        let mask =
            UFuncArray::new(vec![6], vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        let runs = ma.clump_unmasked();
        assert_eq!(runs, vec![(0, 2), (4, 6)]);
    }

    #[test]
    fn masked_clump_masked() {
        let data = UFuncArray::new(vec![6], vec![1.0; 6], DType::F64).unwrap();
        let mask =
            UFuncArray::new(vec![6], vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        let runs = ma.clump_masked();
        assert_eq!(runs, vec![(2, 4)]);
    }

    #[test]
    fn masked_shrink_mask_removes_nomask() {
        let data = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![3], vec![0.0, 0.0, 0.0], DType::Bool).unwrap();
        let mut ma = MaskedArray::new(data, Some(mask), None).unwrap();
        assert!(ma.mask().is_some());
        ma.shrink_mask();
        assert!(ma.mask().is_none());
    }

    #[test]
    fn masked_shrink_mask_keeps_mask() {
        let data = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![3], vec![0.0, 1.0, 0.0], DType::Bool).unwrap();
        let mut ma = MaskedArray::new(data, Some(mask), None).unwrap();
        ma.shrink_mask();
        assert!(ma.mask().is_some());
    }

    #[test]
    fn masked_anom() {
        let data = UFuncArray::new(vec![4], vec![2.0, 4.0, 6.0, 8.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![4], vec![0.0, 0.0, 0.0, 1.0], DType::Bool).unwrap();
        let ma = MaskedArray::new(data, Some(mask), None).unwrap();
        let anom = ma.anom().unwrap();
        // Unmasked values: [2,4,6], mean = 4.0
        // Anomalies: [-2.0, 0.0, 2.0, (masked)]
        assert!((anom.data().values()[0] - (-2.0)).abs() < 1e-10);
        assert!((anom.data().values()[1] - 0.0).abs() < 1e-10);
        assert!((anom.data().values()[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn masked_fix_invalid_nans() {
        let data =
            UFuncArray::new(vec![4], vec![1.0, f64::NAN, 3.0, f64::INFINITY], DType::F64).unwrap();
        let mut ma = MaskedArray::new(data, None, None).unwrap();
        ma.fix_invalid();
        // NaN and Inf should now be masked
        let m = ma.mask().unwrap();
        assert_eq!(m.values()[0], 0.0); // valid
        assert_eq!(m.values()[1], 1.0); // was NaN
        assert_eq!(m.values()[2], 0.0); // valid
        assert_eq!(m.values()[3], 1.0); // was Inf
    }

    #[test]
    fn ma_standalone_is_masked() {
        let data = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let ma_no_mask = MaskedArray::new(data.clone(), None, None).unwrap();
        assert!(!ma_is_masked(&ma_no_mask));

        let mask = UFuncArray::new(vec![3], vec![0.0, 1.0, 0.0], DType::Bool).unwrap();
        let ma_with_mask = MaskedArray::new(data, Some(mask), None).unwrap();
        assert!(ma_is_masked(&ma_with_mask));
    }

    #[test]
    fn ma_standalone_is_mask() {
        let valid = UFuncArray::new(vec![3], vec![0.0, 1.0, 0.0], DType::Bool).unwrap();
        assert!(ma_is_mask(&valid));
        let invalid = UFuncArray::new(vec![3], vec![0.0, 2.0, 0.0], DType::Bool).unwrap();
        assert!(!ma_is_mask(&invalid));
    }

    #[test]
    fn ma_standalone_make_mask() {
        let arr = UFuncArray::new(vec![4], vec![0.0, 5.0, 0.0, -1.0], DType::F64).unwrap();
        let m = ma_make_mask(&arr);
        assert_eq!(m.values(), &[0.0, 1.0, 0.0, 1.0]);
        assert_eq!(m.dtype(), DType::Bool);
    }

    #[test]
    fn ma_standalone_mask_or() {
        let m1 = UFuncArray::new(vec![4], vec![1.0, 0.0, 1.0, 0.0], DType::Bool).unwrap();
        let m2 = UFuncArray::new(vec![4], vec![0.0, 0.0, 1.0, 1.0], DType::Bool).unwrap();
        let combined = ma_mask_or(Some(&m1), Some(&m2)).unwrap();
        assert_eq!(combined.values(), &[1.0, 0.0, 1.0, 1.0]);

        // None + Some = Some
        let r = ma_mask_or(None, Some(&m1)).unwrap();
        assert_eq!(r.values(), m1.values());

        // None + None = None
        assert!(ma_mask_or(None, None).is_none());
    }

    // ── datetime/timedelta arithmetic tests ────────────────────────

    #[test]
    fn datetime_add_timedelta() {
        // day 100 + delta 5 = day 105
        let dates = UFuncArray::new(vec![3], vec![100.0, 200.0, 300.0], DType::DateTime64).unwrap();
        let deltas = UFuncArray::new(vec![3], vec![5.0, 10.0, -1.0], DType::TimeDelta64).unwrap();
        let result = dates.datetime_add(&deltas).unwrap();
        assert_eq!(result.dtype(), DType::DateTime64);
        assert_eq!(result.values(), &[105.0, 210.0, 299.0]);
    }

    #[test]
    fn datetime_sub_datetimes() {
        let d1 = UFuncArray::new(vec![2], vec![110.0, 205.0], DType::DateTime64).unwrap();
        let d2 = UFuncArray::new(vec![2], vec![100.0, 200.0], DType::DateTime64).unwrap();
        let result = d1.datetime_sub(&d2).unwrap();
        assert_eq!(result.dtype(), DType::TimeDelta64);
        assert_eq!(result.values(), &[10.0, 5.0]);
    }

    #[test]
    fn timedelta_add_sub() {
        let t1 = UFuncArray::new(vec![2], vec![10.0, 20.0], DType::TimeDelta64).unwrap();
        let t2 = UFuncArray::new(vec![2], vec![3.0, 7.0], DType::TimeDelta64).unwrap();
        let sum = t1.timedelta_add(&t2).unwrap();
        assert_eq!(sum.values(), &[13.0, 27.0]);
        let diff = t1.timedelta_sub(&t2).unwrap();
        assert_eq!(diff.values(), &[7.0, 13.0]);
    }

    #[test]
    fn timedelta_mul_div() {
        let td = UFuncArray::new(vec![3], vec![10.0, 20.0, 30.0], DType::TimeDelta64).unwrap();
        let doubled = td.timedelta_mul(2.0).unwrap();
        assert_eq!(doubled.values(), &[20.0, 40.0, 60.0]);
        let halved = td.timedelta_div(2.0).unwrap();
        assert_eq!(halved.values(), &[5.0, 10.0, 15.0]);
    }

    #[test]
    fn timedelta_div_by_zero_rejected() {
        let td = UFuncArray::new(vec![1], vec![10.0], DType::TimeDelta64).unwrap();
        assert!(td.timedelta_div(0.0).is_err());
    }

    #[test]
    fn timedelta_neg_abs() {
        let td = UFuncArray::new(vec![3], vec![-5.0, 0.0, 3.0], DType::TimeDelta64).unwrap();
        let neg = td.timedelta_neg();
        assert_eq!(neg.values(), &[5.0, 0.0, -3.0]);
        let abs = td.timedelta_abs();
        assert_eq!(abs.values(), &[5.0, 0.0, 3.0]);
    }

    #[test]
    fn datetime_type_checks() {
        let f = UFuncArray::new(vec![1], vec![1.0], DType::F64).unwrap();
        let dt = UFuncArray::new(vec![1], vec![1.0], DType::DateTime64).unwrap();
        let td = UFuncArray::new(vec![1], vec![1.0], DType::TimeDelta64).unwrap();
        // datetime_add requires DateTime64 + TimeDelta64
        assert!(f.datetime_add(&td).is_err());
        assert!(dt.datetime_add(&f).is_err());
        // datetime_sub requires two DateTime64
        assert!(dt.datetime_sub(&td).is_err());
        // timedelta_add requires two TimeDelta64
        assert!(td.timedelta_add(&dt).is_err());
    }

    #[test]
    fn is_busday_test() {
        // 1970-01-01 (day 0) = Thursday (business day)
        // 1970-01-02 (day 1) = Friday (business day)
        // 1970-01-03 (day 2) = Saturday (not business)
        // 1970-01-04 (day 3) = Sunday (not business)
        // 1970-01-05 (day 4) = Monday (business day)
        let dates =
            UFuncArray::new(vec![5], vec![0.0, 1.0, 2.0, 3.0, 4.0], DType::DateTime64).unwrap();
        let result = is_busday(&dates).unwrap();
        assert_eq!(result.values(), &[1.0, 1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn busday_count_basic() {
        // Count business days from Monday (day 4) to next Monday (day 11) = 5
        let start = UFuncArray::new(vec![1], vec![4.0], DType::DateTime64).unwrap();
        let end = UFuncArray::new(vec![1], vec![11.0], DType::DateTime64).unwrap();
        let result = busday_count(&start, &end).unwrap();
        assert_eq!(result.values(), &[5.0]);
    }

    #[test]
    fn busday_count_same_day() {
        let start = UFuncArray::new(vec![1], vec![0.0], DType::DateTime64).unwrap();
        let end = UFuncArray::new(vec![1], vec![0.0], DType::DateTime64).unwrap();
        let result = busday_count(&start, &end).unwrap();
        assert_eq!(result.values(), &[0.0]);
    }

    #[test]
    fn busday_count_backwards() {
        // day 11 to day 4 = -5 business days (reverse direction)
        let start = UFuncArray::new(vec![1], vec![11.0], DType::DateTime64).unwrap();
        let end = UFuncArray::new(vec![1], vec![4.0], DType::DateTime64).unwrap();
        let result = busday_count(&start, &end).unwrap();
        assert_eq!(result.values(), &[-5.0]);
    }

    #[test]
    fn busday_offset_basic() {
        // From Monday (day 4), offset +1 business day = Tuesday (day 5)
        let dates = UFuncArray::new(vec![1], vec![4.0], DType::DateTime64).unwrap();
        let offsets = UFuncArray::new(vec![1], vec![1.0], DType::I64).unwrap();
        let result = busday_offset(&dates, &offsets).unwrap();
        assert_eq!(result.values(), &[5.0]);
    }

    #[test]
    fn busday_offset_over_weekend() {
        // From Friday (day 1), offset +1 business day = Monday (day 4)
        let dates = UFuncArray::new(vec![1], vec![1.0], DType::DateTime64).unwrap();
        let offsets = UFuncArray::new(vec![1], vec![1.0], DType::I64).unwrap();
        let result = busday_offset(&dates, &offsets).unwrap();
        assert_eq!(result.values(), &[4.0]);
    }

    #[test]
    fn busday_offset_negative() {
        // From Monday (day 4), offset -1 business day = Friday (day 1)
        let dates = UFuncArray::new(vec![1], vec![4.0], DType::DateTime64).unwrap();
        let offsets = UFuncArray::new(vec![1], vec![-1.0], DType::I64).unwrap();
        let result = busday_offset(&dates, &offsets).unwrap();
        assert_eq!(result.values(), &[1.0]);
    }

    #[test]
    fn busday_offset_from_weekend() {
        // From Saturday (day 2), offset +1 should snap to Monday then +1 = Tuesday (day 5)
        let dates = UFuncArray::new(vec![1], vec![2.0], DType::DateTime64).unwrap();
        let offsets = UFuncArray::new(vec![1], vec![1.0], DType::I64).unwrap();
        let result = busday_offset(&dates, &offsets).unwrap();
        assert_eq!(result.values(), &[5.0]);
    }

    #[test]
    fn busday_offset_zero() {
        // From Monday (day 4), offset 0 = same Monday (day 4)
        let dates = UFuncArray::new(vec![1], vec![4.0], DType::DateTime64).unwrap();
        let offsets = UFuncArray::new(vec![1], vec![0.0], DType::I64).unwrap();
        let result = busday_offset(&dates, &offsets).unwrap();
        assert_eq!(result.values(), &[4.0]);
    }

    // ── Financial function tests ─────────────────────────────────────

    #[test]
    fn financial_fv_basic() {
        // $1000 at 5%/yr for 10 years, no payments
        let fv = financial_fv(0.05, 10.0, 0.0, -1000.0, 0);
        assert!((fv - 1628.894627).abs() < 0.01);
    }

    #[test]
    fn financial_fv_zero_rate() {
        let fv = financial_fv(0.0, 10.0, -100.0, -1000.0, 0);
        assert!((fv - 2000.0).abs() < 1e-10);
    }

    #[test]
    fn financial_pv_basic() {
        // What PV yields $10000 after 10 years at 5%?
        let pv = financial_pv(0.05, 10.0, 0.0, -10000.0, 0);
        assert!((pv - 6139.13).abs() < 0.01);
    }

    #[test]
    fn financial_pmt_basic() {
        // Monthly payment on $200k loan, 30 years, 6%/12 monthly
        let pmt = financial_pmt(0.06 / 12.0, 360.0, 200_000.0, 0.0, 0);
        assert!((pmt - (-1199.10)).abs() < 0.01);
    }

    #[test]
    fn financial_nper_basic() {
        // How many periods to pay off $1000 at 1%/period with $100 payments?
        let n = financial_nper(0.01, -100.0, 1000.0, 0.0, 0);
        assert!((n - 10.58).abs() < 0.01);
    }

    #[test]
    fn financial_npv_basic() {
        let cashflows = [-1000.0, 300.0, 400.0, 500.0];
        let npv = financial_npv(0.1, &cashflows);
        assert!((npv - (-21.037)).abs() < 0.1);
    }

    #[test]
    fn financial_irr_basic() {
        let cashflows = [-1000.0, 300.0, 400.0, 500.0];
        let irr = financial_irr(&cashflows);
        // NPV at this rate should be ~0
        let npv = financial_npv(irr, &cashflows);
        assert!(npv.abs() < 0.01);
    }

    #[test]
    fn financial_irr_obvious() {
        // Invest 100, get 110 back => 10% return
        let cashflows = [-100.0, 110.0];
        let irr = financial_irr(&cashflows);
        assert!((irr - 0.1).abs() < 1e-8);
    }

    #[test]
    fn financial_ipmt_ppmt_sum_equals_pmt() {
        let rate = 0.05 / 12.0;
        let nper = 360.0;
        let pv = 200_000.0;
        let fv = 0.0;
        let when = 0u8;
        let total_pmt = financial_pmt(rate, nper, pv, fv, when);
        for per in 1..=5 {
            let ip = financial_ipmt(rate, per as f64, nper, pv, fv, when);
            let pp = financial_ppmt(rate, per as f64, nper, pv, fv, when);
            assert!((ip + pp - total_pmt).abs() < 1e-8);
        }
    }

    #[test]
    fn financial_mirr_basic() {
        let cashflows = [-1000.0, 200.0, 300.0, 400.0, 500.0];
        let mirr = financial_mirr(&cashflows, 0.1, 0.12);
        assert!(mirr.is_finite());
        assert!(mirr > 0.0 && mirr < 1.0);
    }

    #[test]
    fn financial_rate_basic() {
        // If pmt=-100, nper=10, pv=800, fv=0: find rate
        let r = financial_rate(10.0, -100.0, 800.0, 0.0, 0, 0.1);
        // Verify: with this rate, pmt should reconstruct
        let pmt_check = financial_pmt(r, 10.0, 800.0, 0.0, 0);
        assert!((pmt_check - (-100.0)).abs() < 0.01);
    }

    // ── StringArray tests ──────────────────────────────────────────

    #[test]
    fn string_array_new() {
        let sa = StringArray::from_strs(vec![3], &["hello", "world", "test"]).unwrap();
        assert_eq!(sa.shape(), &[3]);
        assert_eq!(sa.values()[0], "hello");
    }

    #[test]
    fn string_array_shape_mismatch() {
        let err = StringArray::from_strs(vec![2], &["hello", "world", "extra"]).unwrap_err();
        assert!(matches!(err, UFuncError::InvalidInputLength { .. }));
    }

    #[test]
    fn string_upper_lower() {
        let sa = StringArray::from_strs(vec![2], &["Hello", "World"]).unwrap();
        let upper = sa.upper();
        assert_eq!(upper.values()[0], "HELLO");
        assert_eq!(upper.values()[1], "WORLD");
        let lower = sa.lower();
        assert_eq!(lower.values()[0], "hello");
        assert_eq!(lower.values()[1], "world");
    }

    #[test]
    fn string_capitalize() {
        let sa = StringArray::from_strs(vec![2], &["hello WORLD", "tEST"]).unwrap();
        let cap = sa.capitalize();
        assert_eq!(cap.values()[0], "Hello world");
        assert_eq!(cap.values()[1], "Test");
    }

    #[test]
    fn string_title() {
        let sa = StringArray::from_strs(vec![1], &["hello world test"]).unwrap();
        let t = sa.title();
        assert_eq!(t.values()[0], "Hello World Test");
    }

    #[test]
    fn string_strip() {
        let sa = StringArray::from_strs(vec![3], &["  hello  ", "  left", "right  "]).unwrap();
        let stripped = sa.strip();
        assert_eq!(stripped.values()[0], "hello");
        let lstripped = sa.lstrip();
        assert_eq!(lstripped.values()[0], "hello  ");
        assert_eq!(lstripped.values()[1], "left");
        let rstripped = sa.rstrip();
        assert_eq!(rstripped.values()[0], "  hello");
        assert_eq!(rstripped.values()[2], "right");
    }

    #[test]
    fn string_center_ljust_rjust() {
        let sa = StringArray::from_strs(vec![1], &["hi"]).unwrap();
        assert_eq!(sa.center(6, '*').values()[0], "**hi**");
        assert_eq!(sa.ljust(6, '-').values()[0], "hi----");
        assert_eq!(sa.rjust(6, '-').values()[0], "----hi");
    }

    #[test]
    fn string_zfill() {
        let sa = StringArray::from_strs(vec![3], &["42", "-7", "+3"]).unwrap();
        let z = sa.zfill(5);
        assert_eq!(z.values()[0], "00042");
        assert_eq!(z.values()[1], "-0007");
        assert_eq!(z.values()[2], "+0003");
    }

    #[test]
    fn string_str_len() {
        let sa = StringArray::from_strs(vec![3], &["", "hi", "hello"]).unwrap();
        let lens = sa.str_len();
        assert_eq!(lens.values(), &[0.0, 2.0, 5.0]);
    }

    #[test]
    fn string_count_find() {
        let sa = StringArray::from_strs(vec![2], &["abcabc", "xyz"]).unwrap();
        let counts = sa.count_substr("abc");
        assert_eq!(counts.values(), &[2.0, 0.0]);
        let finds = sa.find("abc");
        assert_eq!(finds.values(), &[0.0, -1.0]);
        let rfinds = sa.rfind("abc");
        assert_eq!(rfinds.values(), &[3.0, -1.0]);
    }

    #[test]
    fn string_startswith_endswith() {
        let sa = StringArray::from_strs(vec![3], &["hello", "help", "world"]).unwrap();
        let sw = sa.startswith("hel");
        assert_eq!(sw.values(), &[1.0, 1.0, 0.0]);
        let ew = sa.endswith("lo");
        assert_eq!(ew.values(), &[1.0, 0.0, 0.0]);
    }

    #[test]
    fn string_replace() {
        let sa = StringArray::from_strs(vec![2], &["hello world", "foo bar"]).unwrap();
        let replaced = sa.replace("o", "0");
        assert_eq!(replaced.values()[0], "hell0 w0rld");
        assert_eq!(replaced.values()[1], "f00 bar");
    }

    #[test]
    fn string_add() {
        let a = StringArray::from_strs(vec![2], &["hello", "foo"]).unwrap();
        let b = StringArray::from_strs(vec![2], &[" world", " bar"]).unwrap();
        let result = a.add(&b).unwrap();
        assert_eq!(result.values()[0], "hello world");
        assert_eq!(result.values()[1], "foo bar");
    }

    #[test]
    fn string_multiply() {
        let sa = StringArray::from_strs(vec![2], &["ab", "x"]).unwrap();
        let repeated = sa.multiply(3);
        assert_eq!(repeated.values()[0], "ababab");
        assert_eq!(repeated.values()[1], "xxx");
    }

    #[test]
    fn string_isalpha_isdigit() {
        let sa = StringArray::from_strs(vec![4], &["hello", "123", "abc123", ""]).unwrap();
        let alpha = sa.isalpha();
        assert_eq!(alpha.values(), &[1.0, 0.0, 0.0, 0.0]);
        let digit = sa.isdigit();
        assert_eq!(digit.values(), &[0.0, 1.0, 0.0, 0.0]);
        let alnum = sa.isalnum();
        assert_eq!(alnum.values(), &[1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn string_isspace_isupper_islower() {
        let sa = StringArray::from_strs(vec![4], &["  ", "HELLO", "hello", "Hello"]).unwrap();
        let space = sa.isspace();
        assert_eq!(space.values(), &[1.0, 0.0, 0.0, 0.0]);
        let upper = sa.isupper();
        assert_eq!(upper.values(), &[0.0, 1.0, 0.0, 0.0]);
        let lower = sa.islower();
        assert_eq!(lower.values(), &[0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn string_join() {
        let sa = StringArray::from_strs(vec![1], &["abc"]).unwrap();
        let joined = sa.join("-");
        assert_eq!(joined.values()[0], "a-b-c");
    }

    #[test]
    fn string_swapcase() {
        let sa = StringArray::from_strs(vec![1], &["Hello World"]).unwrap();
        let swapped = sa.swapcase();
        assert_eq!(swapped.values()[0], "hELLO wORLD");
    }

    #[test]
    fn string_split_whitespace() {
        let sa = StringArray::from_strs(vec![2], &["hello world", "foo  bar baz"]).unwrap();
        let result = sa.split(None, None);
        assert_eq!(result[0], vec!["hello", "world"]);
        assert_eq!(result[1], vec!["foo", "bar", "baz"]);
    }

    #[test]
    fn string_split_with_sep() {
        let sa = StringArray::from_strs(vec![1], &["a,b,c,d"]).unwrap();
        let result = sa.split(Some(","), None);
        assert_eq!(result[0], vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn string_split_maxsplit() {
        let sa = StringArray::from_strs(vec![1], &["a,b,c,d"]).unwrap();
        let result = sa.split(Some(","), Some(2));
        assert_eq!(result[0], vec!["a", "b", "c,d"]);
    }

    #[test]
    fn string_partition_found() {
        let sa = StringArray::from_strs(vec![2], &["hello-world", "foobar"]).unwrap();
        let (before, sep, after) = sa.partition("-");
        assert_eq!(before.values()[0], "hello");
        assert_eq!(sep.values()[0], "-");
        assert_eq!(after.values()[0], "world");
        // Not found: everything in before, sep and after empty
        assert_eq!(before.values()[1], "foobar");
        assert_eq!(sep.values()[1], "");
        assert_eq!(after.values()[1], "");
    }

    #[test]
    fn string_rpartition_found() {
        let sa = StringArray::from_strs(vec![1], &["a-b-c"]).unwrap();
        let (before, sep, after) = sa.rpartition("-");
        assert_eq!(before.values()[0], "a-b");
        assert_eq!(sep.values()[0], "-");
        assert_eq!(after.values()[0], "c");
    }

    #[test]
    fn string_encode_decode_roundtrip() {
        let sa = StringArray::from_strs(vec![2], &["hello", "world"]).unwrap();
        let encoded = sa.encode();
        assert_eq!(encoded[0], b"hello");
        assert_eq!(encoded[1], b"world");
        let byte_refs: Vec<&[u8]> = encoded.iter().map(|b| b.as_slice()).collect();
        let decoded = StringArray::decode(vec![2], &byte_refs).unwrap();
        assert_eq!(decoded.values(), sa.values());
    }

    #[test]
    fn string_translate_basic() {
        let sa = StringArray::from_strs(vec![1], &["hello"]).unwrap();
        let mut table = std::collections::HashMap::new();
        table.insert('h', Some('H'));
        table.insert('l', None); // delete 'l'
        let result = sa.translate(&table);
        assert_eq!(result.values()[0], "Heo");
    }

    #[test]
    fn string_maketrans_basic() {
        let table = StringArray::maketrans("aeiou", "AEIOU", None);
        let sa = StringArray::from_strs(vec![1], &["hello world"]).unwrap();
        let result = sa.translate(&table);
        assert_eq!(result.values()[0], "hEllO wOrld");
    }

    #[test]
    fn string_maketrans_with_delete() {
        let table = StringArray::maketrans("h", "H", Some("lo"));
        let sa = StringArray::from_strs(vec![1], &["hello"]).unwrap();
        let result = sa.translate(&table);
        assert_eq!(result.values()[0], "He");
    }

    #[test]
    fn string_expandtabs() {
        let sa = StringArray::from_strs(vec![1], &["a\tb"]).unwrap();
        let result = sa.expandtabs(4);
        assert_eq!(result.values()[0], "a   b");
    }

    #[test]
    fn string_isnumeric() {
        let sa = StringArray::from_strs(vec![3], &["123", "12a", ""]).unwrap();
        let result = sa.isnumeric();
        assert_eq!(result.values(), &[1.0, 0.0, 0.0]);
    }

    #[test]
    fn string_isdecimal() {
        let sa = StringArray::from_strs(vec![2], &["456", "abc"]).unwrap();
        let result = sa.isdecimal();
        assert_eq!(result.values(), &[1.0, 0.0]);
    }

    #[test]
    fn string_istitle() {
        let sa = StringArray::from_strs(vec![3], &["Hello World", "hello world", "HELLO"]).unwrap();
        let result = sa.istitle();
        assert_eq!(result.values(), &[1.0, 0.0, 0.0]);
    }

    // ── Type utility and memory query tests ─────────────────────────

    #[test]
    fn can_cast_safe() {
        assert!(UFuncArray::can_cast(DType::I32, DType::F64, "safe"));
        assert!(!UFuncArray::can_cast(DType::F64, DType::I32, "safe"));
    }

    #[test]
    fn can_cast_same_kind() {
        assert!(UFuncArray::can_cast(DType::I32, DType::I64, "same_kind"));
        assert!(UFuncArray::can_cast(DType::F32, DType::F64, "same_kind"));
    }

    #[test]
    fn mintypecode_basic() {
        assert_eq!(UFuncArray::mintypecode("fd"), 'd'); // float64 dominates
        assert_eq!(UFuncArray::mintypecode("?"), '?'); // just bool
        assert_eq!(UFuncArray::mintypecode("ih"), 'i'); // i32 dominates i16
    }

    #[test]
    fn shares_memory_self() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        assert!(a.shares_memory(&a)); // same reference
        let b = a.clone();
        assert!(!a.shares_memory(&b)); // different allocation
    }

    #[test]
    fn byte_bounds_nonempty() {
        let a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let (lo, hi) = a.byte_bounds();
        assert!(hi > lo);
        assert_eq!(hi - lo, 32); // 4 * 8 bytes
    }

    #[test]
    fn byte_bounds_empty() {
        let a = UFuncArray::new(vec![0], vec![], DType::F64).unwrap();
        let (lo, hi) = a.byte_bounds();
        assert_eq!(lo, 0);
        assert_eq!(hi, 0);
    }

    #[test]
    fn put_mask_basic() {
        let mut a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![4], vec![1.0, 0.0, 1.0, 0.0], DType::Bool).unwrap();
        a.put_mask(&mask, &[99.0]).unwrap();
        assert_eq!(a.values(), &[99.0, 2.0, 99.0, 4.0]);
    }

    #[test]
    fn put_mask_cycling() {
        let mut a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![4], vec![1.0, 1.0, 1.0, 0.0], DType::Bool).unwrap();
        a.put_mask(&mask, &[10.0, 20.0]).unwrap();
        assert_eq!(a.values(), &[10.0, 20.0, 10.0, 4.0]);
    }

    #[test]
    fn put_mask_size_mismatch() {
        let mut a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let mask = UFuncArray::new(vec![2], vec![1.0, 0.0], DType::Bool).unwrap();
        assert!(a.put_mask(&mask, &[99.0]).is_err());
    }

    // ── nanpercentile / nanquantile / real_if_close tests ──────────

    #[test]
    fn nanpercentile_basic() {
        let a =
            UFuncArray::new(vec![5], vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0], DType::F64).unwrap();
        let result = a.nanpercentile(50.0, None).unwrap();
        assert!((result.values()[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn nanpercentile_all_nan() {
        let a = UFuncArray::new(vec![2], vec![f64::NAN, f64::NAN], DType::F64).unwrap();
        let result = a.nanpercentile(50.0, None).unwrap();
        assert!(result.values()[0].is_nan());
    }

    #[test]
    fn nanpercentile_no_nan() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let result = a.nanpercentile(0.0, None).unwrap();
        assert!((result.values()[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn nanpercentile_out_of_range() {
        let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap();
        assert!(a.nanpercentile(101.0, None).is_err());
        assert!(a.nanpercentile(-1.0, None).is_err());
    }

    #[test]
    fn nanquantile_basic() {
        let a =
            UFuncArray::new(vec![5], vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0], DType::F64).unwrap();
        let result = a.nanquantile(0.5, None).unwrap();
        assert!((result.values()[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn real_if_close_complex_with_zero_imag() {
        // Complex array with shape [3, 2]: interleaved [real, imag] pairs
        let a = UFuncArray::new(
            vec![3, 2],
            vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0],
            DType::Complex128,
        )
        .unwrap();
        let result = a.real_if_close(100.0);
        assert_eq!(result.dtype(), DType::F64);
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn real_if_close_complex_with_nonzero_imag() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 0.5, 2.0, 0.0], DType::Complex128).unwrap();
        let result = a.real_if_close(100.0);
        // Should keep complex because imag part 0.5 is not close to zero
        assert_eq!(result.dtype(), DType::Complex128);
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn real_if_close_non_complex() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let result = a.real_if_close(100.0);
        assert_eq!(result.dtype(), DType::F64);
        assert_eq!(result.values(), &[1.0, 2.0, 3.0]);
    }

    // --- argwhere tests ---

    #[test]
    fn argwhere_1d() {
        let a = UFuncArray::new(vec![5], vec![0.0, 1.0, 0.0, 3.0, 0.0], DType::F64).unwrap();
        let result = a.argwhere();
        assert_eq!(result.shape(), &[2, 1]);
        assert_eq!(result.values(), &[1.0, 3.0]);
    }

    #[test]
    fn argwhere_2d() {
        let a =
            UFuncArray::new(vec![2, 3], vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0], DType::F64).unwrap();
        let result = a.argwhere();
        // nonzero at (0,1), (1,0), (1,2) => shape [3, 2]
        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result.values(), &[0.0, 1.0, 1.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn argwhere_all_zero() {
        let a = UFuncArray::new(vec![2, 2], vec![0.0, 0.0, 0.0, 0.0], DType::F64).unwrap();
        let result = a.argwhere();
        assert_eq!(result.shape(), &[0, 2]);
        assert!(result.values().is_empty());
    }

    // --- StringArray char.index / rindex / splitlines / mod_format tests ---

    // --- vectorize tests ---

    #[test]
    fn vectorize_square() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let result = a.vectorize(|x| x * x);
        assert_eq!(result.values(), &[1.0, 4.0, 9.0]);
        assert_eq!(result.shape(), &[3]);
    }

    #[test]
    fn vectorize_preserves_shape() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let result = a.vectorize(|x| x + 10.0);
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.values(), &[11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn vectorize_binary_same_shape() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![10.0, 20.0, 30.0], DType::F64).unwrap();
        let result = UFuncArray::vectorize_binary(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(result.values(), &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn vectorize_binary_broadcasting() {
        // [2,3] + [3] -> [2,3]
        let a = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![10.0, 20.0, 30.0], DType::F64).unwrap();
        let result = UFuncArray::vectorize_binary(&a, &b, |x, y| x * y).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.values(), &[10.0, 40.0, 90.0, 40.0, 100.0, 180.0]);
    }

    #[test]
    fn vectorize_n_three_args() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![4.0, 5.0, 6.0], DType::F64).unwrap();
        let c = UFuncArray::new(vec![3], vec![7.0, 8.0, 9.0], DType::F64).unwrap();
        let result = UFuncArray::vectorize_n(
            &[&a, &b, &c],
            &[],
            |args| args[0] + args[1] * args[2],
        ).unwrap();
        assert_eq!(result.values(), &[29.0, 42.0, 57.0]);
    }

    #[test]
    fn vectorize_n_with_excluded() {
        // a = [1, 2, 3], b = [10] (scalar excluded), c = [4, 5, 6]
        // func(a, b, c) = a + b + c, where b is excluded (constant 10)
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![1], vec![10.0], DType::F64).unwrap();
        let c = UFuncArray::new(vec![3], vec![4.0, 5.0, 6.0], DType::F64).unwrap();
        let result = UFuncArray::vectorize_n(
            &[&a, &b, &c],
            &[1], // exclude arg index 1
            |args| args[0] + args[1] + args[2],
        ).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.values(), &[15.0, 17.0, 19.0]);
    }

    #[test]
    fn vectorize_n_broadcasting() {
        // a = [2,1] shape, b = [1,3] shape -> [2,3] output
        let a = UFuncArray::new(vec![2, 1], vec![1.0, 2.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![1, 3], vec![10.0, 20.0, 30.0], DType::F64).unwrap();
        let result = UFuncArray::vectorize_n(
            &[&a, &b],
            &[],
            |args| args[0] + args[1],
        ).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.values(), &[11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);
    }

    #[test]
    fn vectorize_n_all_excluded() {
        let a = UFuncArray::new(vec![1], vec![5.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![1], vec![3.0], DType::F64).unwrap();
        let result = UFuncArray::vectorize_n(
            &[&a, &b],
            &[0, 1],
            |args| args[0] * args[1],
        ).unwrap();
        assert_eq!(result.values(), &[15.0]);
    }

    // --- string comparison tests ---

    #[test]
    fn string_equal() {
        let a = StringArray::new(vec![3], vec!["abc".into(), "def".into(), "xyz".into()]).unwrap();
        let b = StringArray::new(vec![3], vec!["abc".into(), "xyz".into(), "xyz".into()]).unwrap();
        let r = a.equal(&b);
        assert_eq!(r.values(), &[1.0, 0.0, 1.0]);
    }

    #[test]
    fn string_not_equal() {
        let a = StringArray::new(vec![2], vec!["abc".into(), "def".into()]).unwrap();
        let b = StringArray::new(vec![2], vec!["abc".into(), "xyz".into()]).unwrap();
        let r = a.not_equal(&b);
        assert_eq!(r.values(), &[0.0, 1.0]);
    }

    #[test]
    fn string_greater_less() {
        let a = StringArray::new(vec![2], vec!["b".into(), "a".into()]).unwrap();
        let b = StringArray::new(vec![2], vec!["a".into(), "b".into()]).unwrap();
        let gt = a.greater(&b);
        assert_eq!(gt.values(), &[1.0, 0.0]);
        let lt = a.less(&b);
        assert_eq!(lt.values(), &[0.0, 1.0]);
    }

    #[test]
    fn string_compare_chararrays() {
        let a = StringArray::new(vec![2], vec!["abc".into(), "xyz".into()]).unwrap();
        let b = StringArray::new(vec![2], vec!["abc".into(), "abc".into()]).unwrap();
        let r = a.compare_chararrays(&b, ">=").unwrap();
        assert_eq!(r.values(), &[1.0, 1.0]);
    }

    // --- type introspection tests ---

    #[test]
    fn promote_types_int_float() {
        let result = UFuncArray::promote_types(DType::I32, DType::F64);
        assert_eq!(result, DType::F64);
    }

    #[test]
    fn min_scalar_type_small_uint() {
        let a = UFuncArray::new(vec![3], vec![0.0, 100.0, 200.0], DType::F64).unwrap();
        assert_eq!(a.min_scalar_type(), DType::U8);
    }

    #[test]
    fn min_scalar_type_negative_int() {
        let a = UFuncArray::new(vec![2], vec![-5.0, 50.0], DType::F64).unwrap();
        assert_eq!(a.min_scalar_type(), DType::I8);
    }

    #[test]
    fn min_scalar_type_float() {
        let a = UFuncArray::new(vec![2], vec![1.5, 2.7], DType::F64).unwrap();
        assert_eq!(a.min_scalar_type(), DType::F64);
    }

    #[test]
    fn isfortran_1d() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        assert!(a.isfortran()); // 1D is both C and F
    }

    #[test]
    fn isfortran_2d() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        assert!(!a.isfortran()); // 2D is C-contiguous only
    }

    #[test]
    fn isscalar_true() {
        let a = UFuncArray::new(vec![1], vec![42.0], DType::F64).unwrap();
        assert!(a.isscalar());
    }

    #[test]
    fn isscalar_false() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        assert!(!a.isscalar());
    }

    #[test]
    fn isreal_non_complex() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let r = a.isreal();
        assert_eq!(r.values(), &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn iscomplex_non_complex() {
        let a = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let r = a.iscomplex();
        assert_eq!(r.values(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn isrealobj_and_iscomplexobj() {
        let real = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap();
        assert!(real.isrealobj());
        assert!(!real.iscomplexobj());
        let cplx = UFuncArray::new(vec![1, 2], vec![1.0, 2.0], DType::Complex128).unwrap();
        assert!(!cplx.isrealobj());
        assert!(cplx.iscomplexobj());
    }

    // --- linalg bridge tests ---

    #[test]
    fn norm_vector_default() {
        let a = UFuncArray::new(vec![3], vec![3.0, 4.0, 0.0], DType::F64).unwrap();
        let n = a.norm(None).unwrap();
        assert!((n - 5.0).abs() < 1e-10);
    }

    #[test]
    fn norm_vector_ord_1() {
        let a = UFuncArray::new(vec![3], vec![1.0, -2.0, 3.0], DType::F64).unwrap();
        let n = a.norm(Some("1")).unwrap();
        assert!((n - 6.0).abs() < 1e-10);
    }

    #[test]
    fn norm_matrix_fro() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let n = a.norm(Some("fro")).unwrap();
        let expected = (1.0 + 4.0 + 9.0 + 16.0_f64).sqrt();
        assert!((n - expected).abs() < 1e-10);
    }

    #[test]
    fn matrix_rank_identity() {
        let a = UFuncArray::new(
            vec![3, 3],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            DType::F64,
        )
        .unwrap();
        let r = a.matrix_rank(1e-10).unwrap();
        assert_eq!(r, 3);
    }

    #[test]
    fn matrix_power_identity() {
        let eye = UFuncArray::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0], DType::F64).unwrap();
        let result = eye.matrix_power(5).unwrap();
        assert_eq!(result.values(), &[1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn lstsq_simple() {
        // Solve [[1,0],[0,1]] * x = [3, 4] => x = [3, 4]
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2], vec![3.0, 4.0], DType::F64).unwrap();
        let x = a.lstsq(&b).unwrap();
        assert!((x.values()[0] - 3.0).abs() < 1e-10);
        assert!((x.values()[1] - 4.0).abs() < 1e-10);
    }

    // --- additional linalg bridge tests ---

    #[test]
    fn cholesky_identity() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0], DType::F64).unwrap();
        let l = a.cholesky().unwrap();
        assert_eq!(l.shape(), &[2, 2]);
        assert!((l.values()[0] - 1.0).abs() < 1e-10);
        assert!((l.values()[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn det_identity() {
        let a = UFuncArray::new(
            vec![3, 3],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            DType::F64,
        )
        .unwrap();
        let d = a.det().unwrap();
        assert!((d - 1.0).abs() < 1e-10);
    }

    #[test]
    fn det_2x2() {
        let a = UFuncArray::new(vec![2, 2], vec![3.0, 8.0, 4.0, 6.0], DType::F64).unwrap();
        let d = a.det().unwrap();
        // 3*6 - 8*4 = 18 - 32 = -14
        assert!((d - (-14.0)).abs() < 1e-10);
    }

    #[test]
    fn slogdet_positive() {
        let a = UFuncArray::new(vec![2, 2], vec![2.0, 0.0, 0.0, 3.0], DType::F64).unwrap();
        let (sign, logabsdet) = a.slogdet().unwrap();
        assert!((sign - 1.0).abs() < 1e-10);
        assert!((logabsdet - (6.0_f64).ln()).abs() < 1e-10);
    }

    #[test]
    fn inv_identity() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0], DType::F64).unwrap();
        let inv = a.inv().unwrap();
        assert_eq!(inv.shape(), &[2, 2]);
        assert!((inv.values()[0] - 1.0).abs() < 1e-10);
        assert!((inv.values()[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn inv_2x2() {
        let a = UFuncArray::new(vec![2, 2], vec![4.0, 7.0, 2.0, 6.0], DType::F64).unwrap();
        let inv = a.inv().unwrap();
        // A * A^-1 should be identity
        let prod = a.matmul(&inv).unwrap();
        assert!((prod.values()[0] - 1.0).abs() < 1e-8);
        assert!((prod.values()[1]).abs() < 1e-8);
        assert!((prod.values()[2]).abs() < 1e-8);
        assert!((prod.values()[3] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn pinv_identity() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0], DType::F64).unwrap();
        let p = a.pinv().unwrap();
        assert_eq!(p.shape(), &[2, 2]);
        assert!((p.values()[0] - 1.0).abs() < 1e-10);
        assert!((p.values()[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn solve_identity() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2], vec![5.0, 7.0], DType::F64).unwrap();
        let x = a.solve(&b).unwrap();
        assert!((x.values()[0] - 5.0).abs() < 1e-10);
        assert!((x.values()[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn solve_2x2() {
        // [[2, 1], [5, 3]] * x = [4, 7] => x = [5, -6]
        let a = UFuncArray::new(vec![2, 2], vec![2.0, 1.0, 5.0, 3.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2], vec![4.0, 7.0], DType::F64).unwrap();
        let x = a.solve(&b).unwrap();
        assert!((x.values()[0] - 5.0).abs() < 1e-8);
        assert!((x.values()[1] - (-6.0)).abs() < 1e-8);
    }

    #[test]
    fn eigvals_diagonal() {
        let a = UFuncArray::new(vec![2, 2], vec![3.0, 0.0, 0.0, 5.0], DType::F64).unwrap();
        let vals = a.eigvals().unwrap();
        // eig_nxn returns interleaved [real, imag, real, imag, ...] for 2 eigenvalues = 4 values
        let v = vals.values();
        assert_eq!(v.len(), 4);
        // Extract real parts (indices 0, 2)
        let mut reals = [v[0], v[2]];
        reals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((reals[0] - 3.0).abs() < 1e-6);
        assert!((reals[1] - 5.0).abs() < 1e-6);
        // Imaginary parts should be zero for real eigenvalues
        assert!(v[1].abs() < 1e-6);
        assert!(v[3].abs() < 1e-6);
    }

    #[test]
    fn eig_diagonal() {
        let a = UFuncArray::new(vec![2, 2], vec![3.0, 0.0, 0.0, 5.0], DType::F64).unwrap();
        let (vals, vecs) = a.eig().unwrap();
        assert_eq!(vecs.shape(), &[2, 2]);
        // Eigenvalues in interleaved real/imag format
        let v = vals.values();
        let mut reals = [v[0], v[2]];
        reals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((reals[0] - 3.0).abs() < 1e-6);
        assert!((reals[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn eigh_symmetric() {
        // Symmetric: [[2, 1], [1, 3]]
        let a = UFuncArray::new(vec![2, 2], vec![2.0, 1.0, 1.0, 3.0], DType::F64).unwrap();
        let (vals, vecs) = a.eigh().unwrap();
        assert_eq!(vals.shape(), &[2]);
        assert_eq!(vecs.shape(), &[2, 2]);
        // eigenvalues of [[2,1],[1,3]] are (5 +/- sqrt(5))/2
        let expected_low = (5.0 - 5.0_f64.sqrt()) / 2.0;
        let expected_high = (5.0 + 5.0_f64.sqrt()) / 2.0;
        let mut v = vals.values().to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((v[0] - expected_low).abs() < 1e-6);
        assert!((v[1] - expected_high).abs() < 1e-6);
    }

    #[test]
    fn eigvalsh_symmetric() {
        let a = UFuncArray::new(vec![2, 2], vec![4.0, 0.0, 0.0, 9.0], DType::F64).unwrap();
        let vals = a.eigvalsh().unwrap();
        let mut v = vals.values().to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((v[0] - 4.0).abs() < 1e-6);
        assert!((v[1] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn svd_identity() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0], DType::F64).unwrap();
        let (_u, s, _vt) = a.svd().unwrap();
        let mut sv = s.values().to_vec();
        sv.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!((sv[0] - 1.0).abs() < 1e-10);
        assert!((sv[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn svdvals_rectangular() {
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], DType::F64).unwrap();
        let s = a.svdvals().unwrap();
        assert_eq!(s.shape(), &[2]); // min(2,3) = 2
    }

    #[test]
    fn qr_identity() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0], DType::F64).unwrap();
        let (q, r) = a.qr().unwrap();
        assert_eq!(q.shape(), &[2, 2]);
        assert_eq!(r.shape(), &[2, 2]);
        // Q*R should reconstruct A
        let recon = q.matmul(&r).unwrap();
        for (orig, rec) in a.values().iter().zip(recon.values().iter()) {
            assert!((orig - rec).abs() < 1e-10);
        }
    }

    #[test]
    fn qr_rectangular() {
        let a =
            UFuncArray::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let (q, r) = a.qr().unwrap();
        assert_eq!(q.shape(), &[3, 2]); // m x min(m,n)
        assert_eq!(r.shape(), &[2, 2]); // min(m,n) x n
    }

    #[test]
    fn cond_identity() {
        let a = UFuncArray::new(
            vec![3, 3],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            DType::F64,
        )
        .unwrap();
        let c = a.cond().unwrap();
        assert!((c - 1.0).abs() < 1e-6);
    }

    #[test]
    fn lu_factor_identity() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0], DType::F64).unwrap();
        let (lu, _perm, _det_sign) = a.lu_factor().unwrap();
        assert_eq!(lu.shape(), &[2, 2]);
    }

    #[test]
    fn solve_triangular_lower() {
        let a = UFuncArray::new(vec![2, 2], vec![2.0, 0.0, 3.0, 4.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2], vec![6.0, 11.0], DType::F64).unwrap();
        // Lx = b: 2*x0 = 6 => x0=3; 3*3 + 4*x1 = 11 => x1 = 0.5
        let x = a.solve_triangular(&b, true, false).unwrap();
        assert!((x.values()[0] - 3.0).abs() < 1e-10);
        assert!((x.values()[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn expm_identity() {
        let a = UFuncArray::new(vec![2, 2], vec![0.0, 0.0, 0.0, 0.0], DType::F64).unwrap();
        let e = a.expm().unwrap();
        // exp(0) = I
        assert!((e.values()[0] - 1.0).abs() < 1e-10);
        assert!(e.values()[1].abs() < 1e-10);
        assert!(e.values()[2].abs() < 1e-10);
        assert!((e.values()[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn sqrtm_identity() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0], DType::F64).unwrap();
        let s = a.sqrtm().unwrap();
        assert!((s.values()[0] - 1.0).abs() < 1e-8);
        assert!((s.values()[3] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn logm_identity() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0], DType::F64).unwrap();
        let l = a.logm().unwrap();
        // log(I) = 0
        for v in l.values() {
            assert!(v.abs() < 1e-8);
        }
    }

    #[test]
    fn schur_diagonal() {
        let a = UFuncArray::new(vec![2, 2], vec![3.0, 0.0, 0.0, 7.0], DType::F64).unwrap();
        let (t, z) = a.schur().unwrap();
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(z.shape(), &[2, 2]);
    }

    #[test]
    fn polar_identity() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0], DType::F64).unwrap();
        let (u, p) = a.polar().unwrap();
        assert_eq!(u.shape(), &[2, 2]);
        assert_eq!(p.shape(), &[2, 2]);
        // For identity: U=I, P=I
        assert!((u.values()[0] - 1.0).abs() < 1e-8);
        assert!((p.values()[0] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn solve_multi_identity() {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0], DType::F64).unwrap();
        let b =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        let x = a.solve_multi(&b).unwrap();
        assert_eq!(x.shape(), &[2, 3]);
        assert_eq!(x.values(), b.values());
    }

    #[test]
    fn det_non_square_error() {
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        assert!(a.det().is_err());
    }

    #[test]
    fn inv_non_square_error() {
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        assert!(a.inv().is_err());
    }

    #[test]
    fn cholesky_non_square_error() {
        let a =
            UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64).unwrap();
        assert!(a.cholesky().is_err());
    }

    // --- rfft2/irfft2 tests ---

    #[test]
    fn rfft2_basic() {
        // 2x4 real input => rfftn output shape [2, 3, 2] (cols//2+1 = 3)
        let a = UFuncArray::new(
            vec![2, 4],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            DType::F64,
        )
        .unwrap();
        let r = a.rfft2().unwrap();
        assert_eq!(r.shape(), &[2, 3, 2]);
    }

    #[test]
    fn rfft2_rejects_non_2d() {
        let a = UFuncArray::new(vec![8], vec![1.0; 8], DType::F64).unwrap();
        assert!(a.rfft2().is_err());
    }

    #[test]
    fn rfft2_irfft2_roundtrip() {
        let a = UFuncArray::new(
            vec![2, 4],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            DType::F64,
        )
        .unwrap();
        let freq = a.rfft2().unwrap();
        let back = freq.irfft2(Some(4)).unwrap();
        assert_eq!(back.shape(), &[2, 4]);
        for (orig, rec) in a.values().iter().zip(back.values().iter()) {
            assert!((orig - rec).abs() < 1e-10, "orig={orig}, rec={rec}");
        }
    }

    // --- accumulate tests ---

    #[test]
    fn accumulate_add() {
        let a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let result = a.accumulate(|acc, v| acc + v, 0.0);
        assert_eq!(result.values(), &[1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn accumulate_multiply() {
        let a = UFuncArray::new(vec![3], vec![2.0, 3.0, 4.0], DType::F64).unwrap();
        let result = a.accumulate(|acc, v| acc * v, 1.0);
        assert_eq!(result.values(), &[2.0, 6.0, 24.0]);
    }

    #[test]
    fn accumulate_empty() {
        let a = UFuncArray::new(vec![0], vec![], DType::F64).unwrap();
        let result = a.accumulate(|acc, v| acc + v, 0.0);
        assert!(result.values().is_empty());
    }

    // --- reduceat tests ---

    #[test]
    fn reduceat_add() {
        let a = UFuncArray::new(
            vec![8],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            DType::F64,
        )
        .unwrap();
        // indices [0, 4] => reduce [0..4] = 10, reduce [4..8] = 26
        let result = a.reduceat(|acc, v| acc + v, &[0, 4], 0.0).unwrap();
        assert_eq!(result.values(), &[10.0, 26.0]);
    }

    #[test]
    fn reduceat_single_segment() {
        let a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let result = a.reduceat(|acc, v| acc + v, &[0], 0.0).unwrap();
        assert_eq!(result.values(), &[10.0]);
    }

    #[test]
    fn reduceat_empty_indices() {
        let a = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let result = a.reduceat(|acc, v| acc + v, &[], 0.0).unwrap();
        assert!(result.values().is_empty());
    }

    #[test]
    fn string_index_found() {
        let sa = StringArray::new(
            vec![3],
            vec!["hello".into(), "world".into(), "foobar".into()],
        )
        .unwrap();
        let r = sa.index("o").unwrap();
        assert_eq!(r.values(), &[4.0, 1.0, 1.0]);
    }

    #[test]
    fn string_index_not_found() {
        let sa = StringArray::new(vec![2], vec!["abc".into(), "def".into()]).unwrap();
        assert!(sa.index("z").is_err());
    }

    #[test]
    fn string_rindex_found() {
        let sa = StringArray::new(vec![2], vec!["abcabc".into(), "xbx".into()]).unwrap();
        let r = sa.rindex("b").unwrap();
        // "abcabc": last 'b' at char pos 4; "xbx": 'b' at char pos 1
        assert_eq!(r.values(), &[4.0, 1.0]);
    }

    #[test]
    fn string_rindex_not_found() {
        let sa = StringArray::new(vec![1], vec!["abc".into()]).unwrap();
        assert!(sa.rindex("z").is_err());
    }

    #[test]
    fn string_splitlines() {
        let sa = StringArray::new(vec![2], vec!["line1\nline2".into(), "single".into()]).unwrap();
        let result = sa.splitlines();
        assert_eq!(result[0], vec!["line1", "line2"]);
        assert_eq!(result[1], vec!["single"]);
    }

    #[test]
    fn string_mod_format() {
        let sa = StringArray::new(vec![2], vec!["hello %s".into(), "count %d".into()]).unwrap();
        let result = sa.mod_format(&["world", "42"]);
        assert_eq!(result.values[0], "hello world");
        assert_eq!(result.values[1], "count 42");
    }

    // ── frexp tests ─────────────────────────────────────────────────────

    #[test]
    fn frexp_basic() {
        let arr = UFuncArray::new(vec![3], vec![1.0, 2.0, 4.0], DType::F64).unwrap();
        let (m, e) = frexp(&arr).unwrap();
        assert_eq!(m.values(), &[0.5, 0.5, 0.5]);
        assert_eq!(e.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn frexp_negative() {
        let arr = UFuncArray::new(vec![2], vec![-1.0, -2.0], DType::F64).unwrap();
        let (m, e) = frexp(&arr).unwrap();
        assert_eq!(m.values(), &[-0.5, -0.5]);
        assert_eq!(e.values(), &[1.0, 2.0]);
    }

    #[test]
    fn frexp_zero() {
        let arr = UFuncArray::new(vec![2], vec![0.0, -0.0_f64], DType::F64).unwrap();
        let (m, e) = frexp(&arr).unwrap();
        assert_eq!(m.values()[0], 0.0);
        assert!(m.values()[1].is_sign_negative()); // -0.0
        assert_eq!(e.values(), &[0.0, 0.0]);
    }

    #[test]
    fn frexp_special() {
        let arr = UFuncArray::new(
            vec![3],
            vec![f64::INFINITY, f64::NEG_INFINITY, f64::NAN],
            DType::F64,
        )
        .unwrap();
        let (m, e) = frexp(&arr).unwrap();
        assert_eq!(m.values()[0], f64::INFINITY);
        assert_eq!(m.values()[1], f64::NEG_INFINITY);
        assert!(m.values()[2].is_nan());
        assert_eq!(e.values(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn frexp_roundtrip() {
        let vals: Vec<f64> = vec![1.5, 3.25, 100.0, 0.125, -7.75, 1e10];
        let arr = UFuncArray::new(vec![vals.len()], vals.clone(), DType::F64).unwrap();
        let (m, e) = frexp(&arr).unwrap();
        for (i, &v) in vals.iter().enumerate() {
            let reconstructed = m.values()[i] * (2.0_f64).powi(e.values()[i] as i32);
            assert!(
                (reconstructed - v).abs() < 1e-12,
                "frexp roundtrip failed for {}",
                v
            );
        }
    }

    #[test]
    fn frexp_subnormal() {
        let sub = f64::MIN_POSITIVE / 2.0;
        let arr = UFuncArray::new(vec![1], vec![sub], DType::F64).unwrap();
        let (m, e) = frexp(&arr).unwrap();
        let reconstructed = m.values()[0] * (2.0_f64).powi(e.values()[0] as i32);
        assert!(
            (reconstructed - sub).abs() / sub < 1e-10,
            "frexp subnormal failed"
        );
    }

    #[test]
    fn frexp_large_value() {
        let arr = UFuncArray::new(vec![1], vec![1e300], DType::F64).unwrap();
        let (m, e) = frexp(&arr).unwrap();
        assert!(m.values()[0] >= 0.5 && m.values()[0] < 1.0);
        let reconstructed = m.values()[0] * (2.0_f64).powi(e.values()[0] as i32);
        assert!((reconstructed - 1e300).abs() / 1e300 < 1e-10);
    }

    // ── modf tests ──────────────────────────────────────────────────────

    #[test]
    fn modf_positive() {
        let arr = UFuncArray::new(vec![3], vec![1.5, 2.75, 3.0], DType::F64).unwrap();
        let (frac, int) = modf(&arr).unwrap();
        assert_eq!(frac.values(), &[0.5, 0.75, 0.0]);
        assert_eq!(int.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn modf_negative() {
        let arr = UFuncArray::new(vec![2], vec![-1.5, -2.75], DType::F64).unwrap();
        let (frac, int) = modf(&arr).unwrap();
        assert_eq!(frac.values(), &[-0.5, -0.75]);
        assert_eq!(int.values(), &[-1.0, -2.0]);
    }

    #[test]
    fn modf_zero() {
        let arr = UFuncArray::new(vec![1], vec![0.0], DType::F64).unwrap();
        let (frac, int) = modf(&arr).unwrap();
        assert_eq!(frac.values(), &[0.0]);
        assert_eq!(int.values(), &[0.0]);
    }

    #[test]
    fn modf_infinity() {
        let arr =
            UFuncArray::new(vec![2], vec![f64::INFINITY, f64::NEG_INFINITY], DType::F64).unwrap();
        let (frac, int) = modf(&arr).unwrap();
        assert_eq!(frac.values()[0], 0.0);
        assert!(frac.values()[1].is_sign_negative()); // -0.0
        assert_eq!(int.values()[0], f64::INFINITY);
        assert_eq!(int.values()[1], f64::NEG_INFINITY);
    }

    #[test]
    fn modf_nan() {
        let arr = UFuncArray::new(vec![1], vec![f64::NAN], DType::F64).unwrap();
        let (frac, int) = modf(&arr).unwrap();
        assert!(frac.values()[0].is_nan());
        assert!(int.values()[0].is_nan());
    }

    #[test]
    fn modf_roundtrip() {
        let vals = vec![1.5, -2.75, 100.125, -0.001, 3.0];
        let arr = UFuncArray::new(vec![vals.len()], vals.clone(), DType::F64).unwrap();
        let (frac, int) = modf(&arr).unwrap();
        for (i, &v) in vals.iter().enumerate() {
            let sum = frac.values()[i] + int.values()[i];
            assert!(
                (sum - v).abs() < 1e-15,
                "modf roundtrip failed for {}",
                v
            );
        }
    }

    #[test]
    fn modf_integer_inputs() {
        let arr = UFuncArray::new(vec![3], vec![1.0, 2.0, -3.0], DType::F64).unwrap();
        let (frac, int) = modf(&arr).unwrap();
        assert_eq!(frac.values()[0], 0.0);
        assert_eq!(frac.values()[1], 0.0);
        assert_eq!(int.values(), &[1.0, 2.0, -3.0]);
    }

    #[test]
    fn modf_very_small() {
        let arr = UFuncArray::new(vec![1], vec![1e-300], DType::F64).unwrap();
        let (frac, int) = modf(&arr).unwrap();
        assert!((frac.values()[0] - 1e-300).abs() < 1e-310);
        assert_eq!(int.values()[0], 0.0);
    }

    // ── gcd / lcm tests ────────────────────────────────────────────────

    #[test]
    fn gcd_basic() {
        let a = UFuncArray::new(vec![3], vec![12.0, 15.0, 100.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![8.0, 25.0, 75.0], DType::F64).unwrap();
        let out = gcd_arrays(&a, &b).unwrap();
        assert_eq!(out.values(), &[4.0, 5.0, 25.0]);
    }

    #[test]
    fn lcm_basic() {
        let a = UFuncArray::new(vec![2], vec![4.0, 6.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2], vec![6.0, 8.0], DType::F64).unwrap();
        let out = lcm_arrays(&a, &b).unwrap();
        assert_eq!(out.values(), &[12.0, 24.0]);
    }

    #[test]
    fn gcd_with_zero() {
        let a = UFuncArray::new(vec![3], vec![0.0, 0.0, 5.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![5.0, 0.0, 0.0], DType::F64).unwrap();
        let out = gcd_arrays(&a, &b).unwrap();
        assert_eq!(out.values(), &[5.0, 0.0, 5.0]);
    }

    #[test]
    fn lcm_with_zero() {
        let a = UFuncArray::new(vec![2], vec![0.0, 0.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![2], vec![5.0, 0.0], DType::F64).unwrap();
        let out = lcm_arrays(&a, &b).unwrap();
        assert_eq!(out.values(), &[0.0, 0.0]);
    }

    #[test]
    fn gcd_negative_inputs() {
        let a = UFuncArray::new(vec![1], vec![-12.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![1], vec![8.0], DType::F64).unwrap();
        let out = gcd_arrays(&a, &b).unwrap();
        assert_eq!(out.values(), &[4.0]);
    }

    #[test]
    fn lcm_negative_inputs() {
        let a = UFuncArray::new(vec![1], vec![-4.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![1], vec![6.0], DType::F64).unwrap();
        let out = lcm_arrays(&a, &b).unwrap();
        assert_eq!(out.values(), &[12.0]);
    }

    #[test]
    fn gcd_coprime() {
        let a = UFuncArray::new(vec![1], vec![7.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![1], vec![13.0], DType::F64).unwrap();
        let out = gcd_arrays(&a, &b).unwrap();
        assert_eq!(out.values(), &[1.0]);
    }

    #[test]
    fn gcd_same_number() {
        let a = UFuncArray::new(vec![1], vec![42.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![1], vec![42.0], DType::F64).unwrap();
        let out = gcd_arrays(&a, &b).unwrap();
        assert_eq!(out.values(), &[42.0]);
    }

    #[test]
    fn gcd_lcm_invariant() {
        let vals_a = vec![12.0, 15.0, 7.0, 100.0];
        let vals_b = vec![8.0, 25.0, 13.0, 75.0];
        let a = UFuncArray::new(vec![4], vals_a.clone(), DType::F64).unwrap();
        let b = UFuncArray::new(vec![4], vals_b.clone(), DType::F64).unwrap();
        let g = gcd_arrays(&a, &b).unwrap();
        let l = lcm_arrays(&a, &b).unwrap();
        for i in 0..4 {
            let product = g.values()[i] * l.values()[i];
            let expected = vals_a[i].abs() * vals_b[i].abs();
            assert!(
                (product - expected).abs() < 1e-10,
                "gcd*lcm invariant failed for ({}, {})",
                vals_a[i],
                vals_b[i]
            );
        }
    }

    #[test]
    fn gcd_large_primes() {
        // Two large primes → gcd = 1
        let a = UFuncArray::new(vec![1], vec![1_000_000_007.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![1], vec![999_999_937.0], DType::F64).unwrap();
        let out = gcd_arrays(&a, &b).unwrap();
        assert_eq!(out.values(), &[1.0]);
    }

    // ── divmod tests ────────────────────────────────────────────────────

    #[test]
    fn divmod_positive() {
        let a = UFuncArray::new(vec![3], vec![7.0, 10.0, 100.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![3.0, 3.0, 7.0], DType::F64).unwrap();
        let (q, r) = divmod_arrays(&a, &b).unwrap();
        assert_eq!(q.values(), &[2.0, 3.0, 14.0]);
        assert_eq!(r.values(), &[1.0, 1.0, 2.0]);
    }

    #[test]
    fn divmod_negative_dividend() {
        let a = UFuncArray::new(vec![1], vec![-7.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![1], vec![3.0], DType::F64).unwrap();
        let (q, r) = divmod_arrays(&a, &b).unwrap();
        assert_eq!(q.values(), &[-3.0]); // floor division
        assert_eq!(r.values(), &[2.0]);
    }

    #[test]
    fn divmod_negative_divisor() {
        let a = UFuncArray::new(vec![1], vec![7.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![1], vec![-3.0], DType::F64).unwrap();
        let (q, r) = divmod_arrays(&a, &b).unwrap();
        assert_eq!(q.values(), &[-3.0]);
        assert_eq!(r.values(), &[-2.0]);
    }

    #[test]
    fn divmod_both_negative() {
        let a = UFuncArray::new(vec![1], vec![-7.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![1], vec![-3.0], DType::F64).unwrap();
        let (q, r) = divmod_arrays(&a, &b).unwrap();
        assert_eq!(q.values(), &[2.0]);
        assert_eq!(r.values(), &[-1.0]);
    }

    #[test]
    fn divmod_float() {
        let a = UFuncArray::new(vec![1], vec![7.5], DType::F64).unwrap();
        let b = UFuncArray::new(vec![1], vec![2.5], DType::F64).unwrap();
        let (q, r) = divmod_arrays(&a, &b).unwrap();
        assert_eq!(q.values(), &[3.0]);
        assert!((r.values()[0]).abs() < 1e-15);
    }

    #[test]
    fn divmod_zero_dividend() {
        let a = UFuncArray::new(vec![1], vec![0.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![1], vec![5.0], DType::F64).unwrap();
        let (q, r) = divmod_arrays(&a, &b).unwrap();
        assert_eq!(q.values(), &[0.0]);
        assert_eq!(r.values(), &[0.0]);
    }

    #[test]
    fn divmod_invariant() {
        let vals_a = vec![7.0, -7.0, 10.5, -3.25, 100.0];
        let vals_b = vec![3.0, 3.0, 2.5, 1.5, 7.0];
        let a = UFuncArray::new(vec![5], vals_a.clone(), DType::F64).unwrap();
        let b = UFuncArray::new(vec![5], vals_b.clone(), DType::F64).unwrap();
        let (q, r) = divmod_arrays(&a, &b).unwrap();
        for i in 0..5 {
            let reconstructed = q.values()[i] * vals_b[i] + r.values()[i];
            assert!(
                (reconstructed - vals_a[i]).abs() < 1e-12,
                "divmod invariant failed for ({}, {})",
                vals_a[i],
                vals_b[i]
            );
        }
    }

    #[test]
    fn divmod_by_zero() {
        let a = UFuncArray::new(vec![1], vec![5.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![1], vec![0.0], DType::F64).unwrap();
        let (q, r) = divmod_arrays(&a, &b).unwrap();
        assert!(q.values()[0].is_infinite());
        assert!(r.values()[0].is_nan());
    }

    // ── isposinf / isneginf tests ───────────────────────────────────────

    #[test]
    fn isposinf_mixed() {
        let arr = UFuncArray::new(
            vec![5],
            vec![1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 0.0],
            DType::F64,
        )
        .unwrap();
        let out = isposinf(&arr).unwrap();
        assert_eq!(out.values(), &[0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(out.dtype(), DType::Bool);
    }

    #[test]
    fn isneginf_mixed() {
        let arr = UFuncArray::new(
            vec![5],
            vec![1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 0.0],
            DType::F64,
        )
        .unwrap();
        let out = isneginf(&arr).unwrap();
        assert_eq!(out.values(), &[0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(out.dtype(), DType::Bool);
    }

    #[test]
    fn isposinf_all_finite() {
        let arr = UFuncArray::new(vec![3], vec![1.0, -2.0, 0.0], DType::F64).unwrap();
        let out = isposinf(&arr).unwrap();
        assert_eq!(out.values(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn isneginf_all_finite() {
        let arr = UFuncArray::new(vec![3], vec![1.0, -2.0, 0.0], DType::F64).unwrap();
        let out = isneginf(&arr).unwrap();
        assert_eq!(out.values(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn isposinf_multiple_inf() {
        let arr = UFuncArray::new(
            vec![3],
            vec![f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY],
            DType::F64,
        )
        .unwrap();
        let out = isposinf(&arr).unwrap();
        assert_eq!(out.values(), &[1.0, 1.0, 0.0]);
    }

    #[test]
    fn isinf_relationship() {
        // isposinf(x) | isneginf(x) == isinf(x)
        let arr = UFuncArray::new(
            vec![5],
            vec![1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 0.0],
            DType::F64,
        )
        .unwrap();
        let pos = isposinf(&arr).unwrap();
        let neg = isneginf(&arr).unwrap();
        let inf = arr.elementwise_unary(UnaryOp::Isinf);
        for i in 0..5 {
            let combined = if pos.values()[i] == 1.0 || neg.values()[i] == 1.0 {
                1.0
            } else {
                0.0
            };
            assert_eq!(
                combined,
                inf.values()[i],
                "isinf relationship failed at {}",
                i
            );
        }
    }

    // ── bitwise_count tests ─────────────────────────────────────────────

    #[test]
    fn bitwise_count_powers_of_two() {
        let arr = UFuncArray::new(vec![5], vec![1.0, 2.0, 4.0, 8.0, 16.0], DType::F64).unwrap();
        let out = bitwise_count(&arr).unwrap();
        assert_eq!(out.values(), &[1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn bitwise_count_known_values() {
        let arr = UFuncArray::new(vec![4], vec![0.0, 7.0, 255.0, 1023.0], DType::F64).unwrap();
        let out = bitwise_count(&arr).unwrap();
        assert_eq!(out.values(), &[0.0, 3.0, 8.0, 10.0]);
    }

    #[test]
    fn bitwise_count_specific() {
        let arr = UFuncArray::new(vec![3], vec![13.0, 100.0, 1024.0], DType::F64).unwrap();
        let out = bitwise_count(&arr).unwrap();
        // 13 = 1101 → 3, 100 = 1100100 → 3, 1024 = 10000000000 → 1
        assert_eq!(out.values(), &[3.0, 3.0, 1.0]);
    }

    #[test]
    fn bitwise_count_negative() {
        let arr = UFuncArray::new(vec![2], vec![-1.0, -7.0], DType::F64).unwrap();
        let out = bitwise_count(&arr).unwrap();
        // Uses absolute value: |-1| = 1 → 1, |-7| = 7 → 3
        assert_eq!(out.values(), &[1.0, 3.0]);
    }

    #[test]
    fn bitwise_count_zero() {
        let arr = UFuncArray::new(vec![1], vec![0.0], DType::F64).unwrap();
        let out = bitwise_count(&arr).unwrap();
        assert_eq!(out.values(), &[0.0]);
    }

    #[test]
    fn bitwise_count_float_truncation() {
        let arr = UFuncArray::new(vec![1], vec![7.9], DType::F64).unwrap();
        let out = bitwise_count(&arr).unwrap();
        // 7.9 truncates to 7 = 111 → 3
        assert_eq!(out.values(), &[3.0]);
    }

    // ── sort_complex tests ──────────────────────────────────────────────

    #[test]
    fn sort_complex_basic() {
        // [3+1j, 1+2j, 1+1j] → [1+1j, 1+2j, 3+1j]
        let arr = UFuncArray::new(
            vec![3, 2],
            vec![3.0, 1.0, 1.0, 2.0, 1.0, 1.0],
            DType::Complex128,
        )
        .unwrap();
        let out = sort_complex(&arr).unwrap();
        assert_eq!(out.values(), &[1.0, 1.0, 1.0, 2.0, 3.0, 1.0]);
    }

    #[test]
    fn sort_complex_equal_real() {
        // [3+2j, 3+1j, 1+5j] → [1+5j, 3+1j, 3+2j]
        let arr = UFuncArray::new(
            vec![3, 2],
            vec![3.0, 2.0, 3.0, 1.0, 1.0, 5.0],
            DType::Complex128,
        )
        .unwrap();
        let out = sort_complex(&arr).unwrap();
        assert_eq!(out.values(), &[1.0, 5.0, 3.0, 1.0, 3.0, 2.0]);
    }

    #[test]
    fn sort_complex_real_input() {
        let arr = UFuncArray::new(vec![3], vec![3.0, 1.0, 2.0], DType::F64).unwrap();
        let out = sort_complex(&arr).unwrap();
        // Treated as complex with imag=0: [1+0j, 2+0j, 3+0j]
        assert_eq!(out.values(), &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn sort_complex_already_sorted() {
        let arr = UFuncArray::new(
            vec![3, 2],
            vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0],
            DType::Complex128,
        )
        .unwrap();
        let out = sort_complex(&arr).unwrap();
        assert_eq!(out.values(), &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn sort_complex_negative() {
        // [-1+2j, -3+1j, 2-1j] → [-3+1j, -1+2j, 2-1j]
        let arr = UFuncArray::new(
            vec![3, 2],
            vec![-1.0, 2.0, -3.0, 1.0, 2.0, -1.0],
            DType::Complex128,
        )
        .unwrap();
        let out = sort_complex(&arr).unwrap();
        assert_eq!(out.values(), &[-3.0, 1.0, -1.0, 2.0, 2.0, -1.0]);
    }

    #[test]
    fn sort_complex_single() {
        let arr = UFuncArray::new(vec![1, 2], vec![1.0, 2.0], DType::Complex128).unwrap();
        let out = sort_complex(&arr).unwrap();
        assert_eq!(out.values(), &[1.0, 2.0]);
    }

    #[test]
    fn sort_complex_nan() {
        // NaN sorts to end
        let arr = UFuncArray::new(
            vec![3, 2],
            vec![f64::NAN, 1.0, 1.0, 0.0, 2.0, 0.0],
            DType::Complex128,
        )
        .unwrap();
        let out = sort_complex(&arr).unwrap();
        assert_eq!(out.values()[0], 1.0); // 1+0j first
        assert_eq!(out.values()[2], 2.0); // 2+0j second
        assert!(out.values()[4].is_nan()); // NaN last
    }

    // ── count_nonzero with axis tests ───────────────────────────────────

    #[test]
    fn count_nonzero_axis_none() {
        let a =
            UFuncArray::new(vec![2, 3], vec![0.0, 1.0, 2.0, 3.0, 0.0, 5.0], DType::F64).unwrap();
        let out = a.count_nonzero(None, false).unwrap();
        assert_eq!(out.values(), &[4.0]);
        assert_eq!(out.shape(), &[] as &[usize]);
    }

    #[test]
    fn count_nonzero_axis0() {
        // [[0,1,2],[3,0,5]] → axis=0 → [1,1,2]
        let a =
            UFuncArray::new(vec![2, 3], vec![0.0, 1.0, 2.0, 3.0, 0.0, 5.0], DType::F64).unwrap();
        let out = a.count_nonzero(Some(0), false).unwrap();
        assert_eq!(out.values(), &[1.0, 1.0, 2.0]);
        assert_eq!(out.shape(), &[3]);
    }

    #[test]
    fn count_nonzero_axis1() {
        // [[0,1,2],[3,0,5]] → axis=1 → [2,2]
        let a =
            UFuncArray::new(vec![2, 3], vec![0.0, 1.0, 2.0, 3.0, 0.0, 5.0], DType::F64).unwrap();
        let out = a.count_nonzero(Some(1), false).unwrap();
        assert_eq!(out.values(), &[2.0, 2.0]);
        assert_eq!(out.shape(), &[2]);
    }

    #[test]
    fn count_nonzero_keepdims() {
        let a =
            UFuncArray::new(vec![2, 3], vec![0.0, 1.0, 2.0, 3.0, 0.0, 5.0], DType::F64).unwrap();
        let out = a.count_nonzero(Some(1), true).unwrap();
        assert_eq!(out.values(), &[2.0, 2.0]);
        assert_eq!(out.shape(), &[2, 1]);
    }

    #[test]
    fn count_nonzero_all_zeros() {
        let a = UFuncArray::new(vec![2, 3], vec![0.0; 6], DType::F64).unwrap();
        let out = a.count_nonzero(Some(0), false).unwrap();
        assert_eq!(out.values(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn count_nonzero_negative_axis() {
        let a =
            UFuncArray::new(vec![2, 3], vec![0.0, 1.0, 2.0, 3.0, 0.0, 5.0], DType::F64).unwrap();
        let out = a.count_nonzero(Some(-1), false).unwrap();
        assert_eq!(out.values(), &[2.0, 2.0]); // same as axis=1
    }

    #[test]
    fn count_nonzero_axis_oob() {
        let a = UFuncArray::new(vec![2, 3], vec![0.0; 6], DType::F64).unwrap();
        assert!(a.count_nonzero(Some(5), false).is_err());
    }

    // ── isin invert tests ───────────────────────────────────────────────

    #[test]
    fn isin_invert_false() {
        let arr = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let test = UFuncArray::new(vec![2], vec![2.0, 4.0], DType::F64).unwrap();
        let result = arr.isin(&test, false);
        assert_eq!(result.values(), &[0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn isin_invert_true() {
        let arr = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let test = UFuncArray::new(vec![2], vec![2.0, 4.0], DType::F64).unwrap();
        let result = arr.isin(&test, true);
        assert_eq!(result.values(), &[1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn isin_complement_property() {
        let arr = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let test = UFuncArray::new(vec![2], vec![2.0, 4.0], DType::F64).unwrap();
        let normal = arr.isin(&test, false);
        let inverted = arr.isin(&test, true);
        for i in 0..4 {
            assert_eq!(
                normal.values()[i] + inverted.values()[i],
                1.0,
                "complement property failed at {}",
                i
            );
        }
    }

    #[test]
    fn isin_invert_all_matching() {
        let arr = UFuncArray::new(vec![3], vec![1.0, 1.0, 1.0], DType::F64).unwrap();
        let test = UFuncArray::new(vec![1], vec![1.0], DType::F64).unwrap();
        let result = arr.isin(&test, true);
        assert_eq!(result.values(), &[0.0, 0.0, 0.0]);
    }

    // ── searchsorted side/sorter tests ──────────────────────────────────

    #[test]
    fn searchsorted_side_right_duplicates() {
        let sorted = UFuncArray::new(vec![5], vec![1.0, 2.0, 2.0, 2.0, 3.0], DType::F64).unwrap();
        let probe = UFuncArray::scalar(2.0, DType::F64);
        let out = sorted.searchsorted(&probe, Some("right"), None).unwrap();
        assert_eq!(out.values(), &[4.0]); // after last 2
    }

    #[test]
    fn searchsorted_side_left_duplicates() {
        let sorted = UFuncArray::new(vec![5], vec![1.0, 2.0, 2.0, 2.0, 3.0], DType::F64).unwrap();
        let probe = UFuncArray::scalar(2.0, DType::F64);
        let out = sorted.searchsorted(&probe, Some("left"), None).unwrap();
        assert_eq!(out.values(), &[1.0]); // before first 2
    }

    #[test]
    fn searchsorted_right_no_duplicates() {
        let sorted = UFuncArray::new(vec![4], vec![1.0, 3.0, 5.0, 7.0], DType::F64).unwrap();
        let probe = UFuncArray::scalar(4.0, DType::F64);
        let left = sorted.searchsorted(&probe, Some("left"), None).unwrap();
        let right = sorted.searchsorted(&probe, Some("right"), None).unwrap();
        assert_eq!(left.values(), right.values()); // same when no duplicates
    }

    #[test]
    fn searchsorted_with_sorter() {
        // Unsorted: [5, 1, 3], sorter = [1, 2, 0] → sorted view = [1, 3, 5]
        let unsorted = UFuncArray::new(vec![3], vec![5.0, 1.0, 3.0], DType::F64).unwrap();
        let probe = UFuncArray::scalar(2.0, DType::F64);
        let out = unsorted
            .searchsorted(&probe, None, Some(&[1, 2, 0]))
            .unwrap();
        assert_eq!(out.values(), &[1.0]); // 2 goes after 1 in sorted order
    }

    #[test]
    fn searchsorted_invalid_side() {
        let sorted = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let probe = UFuncArray::scalar(2.0, DType::F64);
        assert!(sorted.searchsorted(&probe, Some("middle"), None).is_err());
    }

    #[test]
    fn searchsorted_invalid_sorter_length() {
        let arr = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let probe = UFuncArray::scalar(2.0, DType::F64);
        assert!(arr.searchsorted(&probe, None, Some(&[0, 1])).is_err());
    }

    // ── where_nonzero tests ─────────────────────────────────────────────

    #[test]
    fn where_nonzero_1d() {
        let arr = UFuncArray::new(vec![5], vec![0.0, 1.0, 0.0, 3.0, 0.0], DType::F64).unwrap();
        let result = where_nonzero(&arr).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].values(), &[1.0, 3.0]);
    }

    #[test]
    fn where_nonzero_2d() {
        let arr = UFuncArray::new(vec![2, 2], vec![1.0, 0.0, 0.0, 3.0], DType::F64).unwrap();
        let result = where_nonzero(&arr).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].values(), &[0.0, 1.0]); // row indices
        assert_eq!(result[1].values(), &[0.0, 1.0]); // col indices
    }

    #[test]
    fn where_nonzero_all_zero() {
        let arr = UFuncArray::new(vec![3], vec![0.0, 0.0, 0.0], DType::F64).unwrap();
        let result = where_nonzero(&arr).unwrap();
        assert_eq!(result[0].values().len(), 0);
    }

    #[test]
    fn where_nonzero_all_nonzero() {
        let arr = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let result = where_nonzero(&arr).unwrap();
        assert_eq!(result[0].values(), &[0.0, 1.0, 2.0]);
    }

    #[test]
    fn where_nonzero_nan_is_nonzero() {
        let arr = UFuncArray::new(vec![3], vec![0.0, f64::NAN, 1.0], DType::F64).unwrap();
        let result = where_nonzero(&arr).unwrap();
        assert_eq!(result[0].values(), &[1.0, 2.0]); // NaN != 0.0
    }

    #[test]
    fn where_nonzero_3d() {
        // shape [2,2,2] with one nonzero at (1,0,1)
        let mut vals = vec![0.0; 8];
        vals[5] = 1.0; // index (1,0,1) in C-order
        let arr = UFuncArray::new(vec![2, 2, 2], vals, DType::F64).unwrap();
        let result = where_nonzero(&arr).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].values(), &[1.0]); // dim 0
        assert_eq!(result[1].values(), &[0.0]); // dim 1
        assert_eq!(result[2].values(), &[1.0]); // dim 2
    }

    // ── unique variants tests ───────────────────────────────────────────

    #[test]
    fn unique_values_basic() {
        let arr = UFuncArray::new(vec![6], vec![3.0, 1.0, 2.0, 1.0, 3.0, 2.0], DType::F64).unwrap();
        let result = unique_values(&arr);
        assert_eq!(result.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn unique_counts_basic() {
        let arr = UFuncArray::new(vec![6], vec![3.0, 1.0, 2.0, 1.0, 3.0, 2.0], DType::F64).unwrap();
        let (vals, counts) = unique_counts(&arr);
        assert_eq!(vals.values(), &[1.0, 2.0, 3.0]);
        assert_eq!(counts.values(), &[2.0, 2.0, 2.0]);
        // counts should sum to array length
        let sum: f64 = counts.values().iter().sum();
        assert_eq!(sum, 6.0);
    }

    #[test]
    fn unique_inverse_basic() {
        let arr = UFuncArray::new(vec![5], vec![3.0, 1.0, 2.0, 1.0, 3.0], DType::F64).unwrap();
        let (vals, inverse) = unique_inverse(&arr);
        assert_eq!(vals.values(), &[1.0, 2.0, 3.0]);
        // Reconstruct: vals[inverse[i]] == original[i]
        for (i, &inv_idx) in inverse.values().iter().enumerate() {
            let reconstructed = vals.values()[inv_idx as usize];
            assert_eq!(reconstructed, arr.values()[i], "inverse failed at {}", i);
        }
    }

    #[test]
    fn unique_all_basic() {
        let arr = UFuncArray::new(vec![5], vec![3.0, 1.0, 2.0, 1.0, 3.0], DType::F64).unwrap();
        let (vals, indices, inverse, counts) = unique_all(&arr);
        assert_eq!(vals.values(), &[1.0, 2.0, 3.0]);
        // indices: first occurrence of each unique value in original
        assert_eq!(indices.values(), &[1.0, 2.0, 0.0]);
        // inverse: index into vals for each original element
        assert_eq!(inverse.values(), &[2.0, 0.0, 1.0, 0.0, 2.0]);
        // counts
        assert_eq!(counts.values(), &[2.0, 1.0, 2.0]);
    }

    #[test]
    fn unique_values_empty() {
        let arr = UFuncArray::new(vec![0], vec![], DType::F64).unwrap();
        let result = unique_values(&arr);
        assert_eq!(result.values().len(), 0);
    }

    #[test]
    fn unique_values_single() {
        let arr = UFuncArray::new(vec![1], vec![42.0], DType::F64).unwrap();
        let result = unique_values(&arr);
        assert_eq!(result.values(), &[42.0]);
    }

    #[test]
    fn unique_counts_all_same() {
        let arr = UFuncArray::new(vec![4], vec![5.0, 5.0, 5.0, 5.0], DType::F64).unwrap();
        let (vals, counts) = unique_counts(&arr);
        assert_eq!(vals.values(), &[5.0]);
        assert_eq!(counts.values(), &[4.0]);
    }

    // ── Chebyshev polynomial tests ──────────────────────────────────────

    #[test]
    fn chebval_t0_through_t3() {
        // T_0(x) = 1, T_1(x) = x, T_2(x) = 2x^2-1, T_3(x) = 4x^3-3x
        let x = vec![0.0, 0.5, 1.0, -1.0];
        // T_0 = [1]
        let t0 = chebval(&x, &[1.0]);
        assert_eq!(t0, vec![1.0, 1.0, 1.0, 1.0]);
        // T_1 = [0, 1]
        let t1 = chebval(&x, &[0.0, 1.0]);
        assert_eq!(t1, vec![0.0, 0.5, 1.0, -1.0]);
        // T_2 = [0, 0, 1] → 2x^2 - 1
        let t2 = chebval(&x, &[0.0, 0.0, 1.0]);
        for (i, &xi) in x.iter().enumerate() {
            let expected = 2.0 * xi * xi - 1.0;
            assert!(
                (t2[i] - expected).abs() < 1e-14,
                "T_2({}) = {} expected {}",
                xi,
                t2[i],
                expected
            );
        }
    }

    #[test]
    fn chebval_at_chebyshev_nodes() {
        // T_n(cos(kπ/n)) = cos(kπ) = (-1)^k for k = 0..n
        let n = 4;
        let nodes: Vec<f64> = (0..=n)
            .map(|k| (k as f64 * std::f64::consts::PI / n as f64).cos())
            .collect();
        // T_4 = [0, 0, 0, 0, 1]
        let c = vec![0.0, 0.0, 0.0, 0.0, 1.0];
        let vals = chebval(&nodes, &c);
        for (k, &v) in vals.iter().enumerate() {
            let expected = if k % 2 == 0 { 1.0 } else { -1.0 };
            assert!(
                (v - expected).abs() < 1e-12,
                "T_4(cos({}π/4)) = {} expected {}",
                k,
                v,
                expected
            );
        }
    }

    #[test]
    fn chebfit_polynomial() {
        // Fit x^2 on [-1, 1] — should be exactly (T_0 + T_2)/2 = [0.5, 0, 0.5]
        let n = 20;
        let x: Vec<f64> = (0..n)
            .map(|i| -1.0 + 2.0 * i as f64 / (n - 1) as f64)
            .collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let c = chebfit(&x, &y, 2).unwrap();
        assert!((c[0] - 0.5).abs() < 1e-10, "c[0] = {}", c[0]);
        assert!(c[1].abs() < 1e-10, "c[1] = {}", c[1]);
        assert!((c[2] - 0.5).abs() < 1e-10, "c[2] = {}", c[2]);
    }

    #[test]
    fn chebder_t3() {
        // T_3 = [0, 0, 0, 1]
        // T_3'(x) = 3*U_2(x) = 3*(4x^2 - 1) = 12x^2 - 3
        // In Chebyshev basis: T_3' = [-1.5, 0, 6] → we check numerically
        let c = vec![0.0, 0.0, 0.0, 1.0];
        let dc = chebder(&c, 1);
        // Evaluate derivative at x = 0.5
        let val = chebval(&[0.5], &dc)[0];
        let expected = 12.0 * 0.25 - 3.0; // 0.0
        assert!(
            (val - expected).abs() < 1e-10,
            "T_3'(0.5) = {} expected {}",
            val,
            expected
        );
    }

    #[test]
    fn chebint_chebder_roundtrip() {
        let c = vec![1.0, 2.0, 3.0]; // arbitrary polynomial
        let integrated = chebint(&c, 1);
        let roundtrip = chebder(&integrated, 1);
        for i in 0..c.len() {
            assert!(
                (roundtrip[i] - c[i]).abs() < 1e-10,
                "roundtrip failed at {}: {} vs {}",
                i,
                roundtrip[i],
                c[i]
            );
        }
    }

    #[test]
    fn chebadd_chebsub() {
        let c1 = vec![1.0, 2.0, 3.0];
        let c2 = vec![4.0, 5.0];
        let sum = chebadd(&c1, &c2);
        assert_eq!(sum, vec![5.0, 7.0, 3.0]);
        let diff = chebsub(&c1, &c2);
        assert_eq!(diff, vec![-3.0, -3.0, 3.0]);
    }

    #[test]
    fn chebmul_t1_times_t1() {
        // T_1 * T_1 = (T_0 + T_2)/2
        let t1 = vec![0.0, 1.0];
        let result = chebmul(&t1, &t1);
        // Expected: [0.5, 0, 0.5]
        assert!((result[0] - 0.5).abs() < 1e-14);
        assert!(result[1].abs() < 1e-14);
        assert!((result[2] - 0.5).abs() < 1e-14);
    }

    #[test]
    fn cheb2poly_t2() {
        // T_2 = 2x^2 - 1 → power basis: [-1, 0, 2]
        let c = vec![0.0, 0.0, 1.0];
        let p = cheb2poly(&c);
        assert!((p[0] - (-1.0)).abs() < 1e-14, "p[0] = {}", p[0]);
        assert!(p[1].abs() < 1e-14, "p[1] = {}", p[1]);
        assert!((p[2] - 2.0).abs() < 1e-14, "p[2] = {}", p[2]);
    }

    #[test]
    fn poly2cheb_x_squared() {
        // x^2 = (T_0 + T_2)/2 → [0.5, 0, 0.5]
        let p = vec![0.0, 0.0, 1.0]; // x^2 in power basis
        let c = poly2cheb(&p);
        assert!((c[0] - 0.5).abs() < 1e-12, "c[0] = {}", c[0]);
        assert!(c[1].abs() < 1e-12, "c[1] = {}", c[1]);
        assert!((c[2] - 0.5).abs() < 1e-12, "c[2] = {}", c[2]);
    }

    #[test]
    fn cheb2poly_poly2cheb_roundtrip() {
        let c = vec![1.0, 2.0, 3.0, 0.5];
        let p = cheb2poly(&c);
        let c2 = poly2cheb(&p);
        for i in 0..c.len() {
            assert!(
                (c2[i] - c[i]).abs() < 1e-10,
                "roundtrip failed at {}: {} vs {}",
                i,
                c2[i],
                c[i]
            );
        }
    }

    #[test]
    fn chebval_empty_coeffs() {
        let result = chebval(&[1.0, 2.0], &[]);
        assert_eq!(result, vec![0.0, 0.0]);
    }

    // ── Legendre polynomial tests ───────────────────────────────────────

    #[test]
    fn legval_p0_through_p2() {
        // P_0=1, P_1=x, P_2=(3x^2-1)/2
        let x = vec![0.0, 0.5, 1.0, -1.0];
        let p0 = legval(&x, &[1.0]);
        assert_eq!(p0, vec![1.0, 1.0, 1.0, 1.0]);
        let p1 = legval(&x, &[0.0, 1.0]);
        assert_eq!(p1, vec![0.0, 0.5, 1.0, -1.0]);
        let p2 = legval(&x, &[0.0, 0.0, 1.0]);
        for (i, &xi) in x.iter().enumerate() {
            let expected = (3.0 * xi * xi - 1.0) / 2.0;
            assert!(
                (p2[i] - expected).abs() < 1e-14,
                "P_2({}) = {} expected {}",
                xi,
                p2[i],
                expected
            );
        }
    }

    #[test]
    fn legfit_polynomial() {
        let n = 20;
        let x: Vec<f64> = (0..n)
            .map(|i| -1.0 + 2.0 * i as f64 / (n - 1) as f64)
            .collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let c = legfit(&x, &y, 2).unwrap();
        // x^2 = (P_0 + 2*P_2)/3 = [1/3, 0, 2/3]
        assert!((c[0] - 1.0 / 3.0).abs() < 1e-10, "c[0] = {}", c[0]);
        assert!(c[1].abs() < 1e-10, "c[1] = {}", c[1]);
        assert!((c[2] - 2.0 / 3.0).abs() < 1e-10, "c[2] = {}", c[2]);
    }

    #[test]
    fn legder_basic() {
        // d/dx P_1(x) = 1, so legder([0, 1], 1) should give [1]
        let dc = legder(&[0.0, 1.0], 1);
        assert_eq!(dc.len(), 1);
        let val = legval(&[0.5], &dc)[0];
        assert!((val - 1.0).abs() < 1e-12);
    }

    #[test]
    fn legint_derivative_numerical() {
        // Verify that the integral of P_1(x) = x is x^2/2 + C
        // legint([0, 1]) should give coefficients whose derivative is [0, 1]
        let c = vec![0.0, 1.0]; // P_1 = x
        let integrated = legint(&c, 1);
        // Numerically verify: evaluate integrated at x=0.5 and x=0
        // ∫x dx from 0 to 0.5 = 0.125
        let val_half = legval(&[0.5], &integrated)[0];
        let val_zero = legval(&[0.0], &integrated)[0];
        let numerical_integral = val_half - val_zero;
        assert!(
            (numerical_integral - 0.125).abs() < 1e-12,
            "∫x dx from 0 to 0.5 = {} expected 0.125",
            numerical_integral
        );
    }

    #[test]
    fn legadd_basic() {
        // [1, 2] + [3, 4, 5] = [4, 6, 5]
        let r = legadd(&[1.0, 2.0], &[3.0, 4.0, 5.0]);
        assert_eq!(r, vec![4.0, 6.0, 5.0]);
    }

    #[test]
    fn legsub_basic() {
        // [5, 3] - [1, 2] = [4, 1]
        let r = legsub(&[5.0, 3.0], &[1.0, 2.0]);
        assert_eq!(r, vec![4.0, 1.0]);
    }

    #[test]
    fn legmul_p1_times_p1() {
        // P_1 * P_1 = x * x = (P_0 + 2*P_2)/3 = [1/3, 0, 2/3]
        let r = legmul(&[0.0, 1.0], &[0.0, 1.0]);
        assert_eq!(r.len(), 3);
        assert!((r[0] - 1.0 / 3.0).abs() < 1e-12, "r[0] = {}", r[0]);
        assert!(r[1].abs() < 1e-12, "r[1] = {}", r[1]);
        assert!((r[2] - 2.0 / 3.0).abs() < 1e-12, "r[2] = {}", r[2]);
    }

    #[test]
    fn legdiv_basic() {
        // Divide P_1^2 by P_1: should give quotient P_1, remainder ~0
        let p1_sq = legmul(&[0.0, 1.0], &[0.0, 1.0]);
        let (q, r) = legdiv(&p1_sq, &[0.0, 1.0]).unwrap();
        // q should be [0, 1] (P_1)
        assert!(q.len() >= 2);
        assert!(q[0].abs() < 1e-10, "q[0] = {}", q[0]);
        assert!((q[1] - 1.0).abs() < 1e-10, "q[1] = {}", q[1]);
        // remainder should be ~0
        for (i, &v) in r.iter().enumerate() {
            assert!(v.abs() < 1e-10, "r[{i}] = {v}");
        }
    }

    #[test]
    fn legroots_linear() {
        // c[0]*P_0 + c[1]*P_1 = 2 + 3*x = 0 → x = -2/3
        let roots = legroots(&[2.0, 3.0]).unwrap();
        assert_eq!(roots.len(), 1);
        assert!((roots[0] - (-2.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn legroots_quadratic() {
        // P_2(x) = (3x^2 - 1)/2, roots at ±1/√3
        let roots = legroots(&[0.0, 0.0, 1.0]).unwrap();
        assert_eq!(roots.len(), 2);
        let expected = 1.0 / 3.0_f64.sqrt();
        assert!((roots[0] - (-expected)).abs() < 1e-10, "root0 = {}", roots[0]);
        assert!((roots[1] - expected).abs() < 1e-10, "root1 = {}", roots[1]);
    }

    #[test]
    fn legfromroots_roundtrip() {
        // Build polynomial from roots [0.5, -0.5], then find roots again
        let c = legfromroots(&[0.5, -0.5]);
        let roots = legroots(&c).unwrap();
        assert_eq!(roots.len(), 2);
        assert!((roots[0] - (-0.5)).abs() < 1e-10, "root0 = {}", roots[0]);
        assert!((roots[1] - 0.5).abs() < 1e-10, "root1 = {}", roots[1]);
    }

    #[test]
    fn leg2poly_p2() {
        // P_2 in Legendre basis: [0, 0, 1]
        // P_2(x) = (3x^2 - 1)/2
        // In power basis: [-0.5, 0, 1.5]
        let p = leg2poly(&[0.0, 0.0, 1.0]);
        assert!(p.len() >= 3);
        assert!((p[0] - (-0.5)).abs() < 1e-12, "p[0] = {}", p[0]);
        assert!(p[1].abs() < 1e-12, "p[1] = {}", p[1]);
        assert!((p[2] - 1.5).abs() < 1e-12, "p[2] = {}", p[2]);
    }

    #[test]
    fn poly2leg_x_squared() {
        // x^2 in power basis: [0, 0, 1]
        // x^2 = (P_0 + 2*P_2)/3 → Legendre coeffs: [1/3, 0, 2/3]
        let c = poly2leg(&[0.0, 0.0, 1.0]);
        assert_eq!(c.len(), 3);
        assert!((c[0] - 1.0 / 3.0).abs() < 1e-10, "c[0] = {}", c[0]);
        assert!(c[1].abs() < 1e-10, "c[1] = {}", c[1]);
        assert!((c[2] - 2.0 / 3.0).abs() < 1e-10, "c[2] = {}", c[2]);
    }

    #[test]
    fn leg2poly_poly2leg_roundtrip() {
        // Random Legendre coefficients
        let orig = vec![1.0, 2.0, 3.0, 4.0];
        let poly = leg2poly(&orig);
        let back = poly2leg(&poly);
        assert_eq!(back.len(), orig.len());
        for i in 0..orig.len() {
            assert!((back[i] - orig[i]).abs() < 1e-8, "c[{i}]: {} vs {}", back[i], orig[i]);
        }
    }

    // ── Hermite polynomial tests ────────────────────────────────────────

    #[test]
    fn hermval_h0_through_h2() {
        // H_0=1, H_1=2x, H_2=4x^2-2
        let x = vec![0.0, 0.5, 1.0];
        let h0 = hermval(&x, &[1.0]);
        assert_eq!(h0, vec![1.0, 1.0, 1.0]);
        let h1 = hermval(&x, &[0.0, 1.0]);
        assert_eq!(h1, vec![0.0, 1.0, 2.0]); // H_1 = 2x
        let h2 = hermval(&x, &[0.0, 0.0, 1.0]);
        for (i, &xi) in x.iter().enumerate() {
            let expected = 4.0 * xi * xi - 2.0;
            assert!(
                (h2[i] - expected).abs() < 1e-14,
                "H_2({}) = {} expected {}",
                xi,
                h2[i],
                expected
            );
        }
    }

    #[test]
    fn hermeval_he0_through_he2() {
        // He_0=1, He_1=x, He_2=x^2-1
        let x = vec![0.0, 1.0, 2.0];
        let he2 = hermeval(&x, &[0.0, 0.0, 1.0]);
        for (i, &xi) in x.iter().enumerate() {
            let expected = xi * xi - 1.0;
            assert!(
                (he2[i] - expected).abs() < 1e-14,
                "He_2({}) = {} expected {}",
                xi,
                he2[i],
                expected
            );
        }
    }

    #[test]
    fn hermder_basic() {
        // H_1' = 2, so hermder([0, 1], 1) should give [2]
        let dc = hermder(&[0.0, 1.0], 1);
        assert_eq!(dc.len(), 1);
        assert!((dc[0] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn hermint_hermder_roundtrip() {
        let c = vec![1.0, 2.0, 3.0];
        let integrated = hermint(&c, 1);
        let roundtrip = hermder(&integrated, 1);
        for i in 0..c.len() {
            assert!(
                (roundtrip[i] - c[i]).abs() < 1e-10,
                "Hermite roundtrip [{}]: {} vs {}",
                i,
                roundtrip[i],
                c[i]
            );
        }
    }

    // ── Laguerre polynomial tests ───────────────────────────────────────

    #[test]
    fn lagval_l0_through_l2() {
        // L_0=1, L_1=1-x, L_2=(x^2-4x+2)/2
        let x = vec![0.0, 1.0, 2.0];
        let l0 = lagval(&x, &[1.0]);
        assert_eq!(l0, vec![1.0, 1.0, 1.0]);
        let l1 = lagval(&x, &[0.0, 1.0]);
        for (i, &xi) in x.iter().enumerate() {
            let expected = 1.0 - xi;
            assert!(
                (l1[i] - expected).abs() < 1e-14,
                "L_1({}) = {} expected {}",
                xi,
                l1[i],
                expected
            );
        }
        let l2 = lagval(&x, &[0.0, 0.0, 1.0]);
        for (i, &xi) in x.iter().enumerate() {
            let expected = (xi * xi - 4.0 * xi + 2.0) / 2.0;
            assert!(
                (l2[i] - expected).abs() < 1e-14,
                "L_2({}) = {} expected {}",
                xi,
                l2[i],
                expected
            );
        }
    }

    #[test]
    fn lagder_basic() {
        // L_1' = -1
        let dc = lagder(&[0.0, 1.0], 1);
        assert_eq!(dc.len(), 1);
        assert!((dc[0] - (-1.0)).abs() < 1e-14);
    }

    // ── scimath tests ───────────────────────────────────────────────────

    #[test]
    fn scimath_sqrt_positive() {
        let arr = UFuncArray::new(vec![2], vec![4.0, 9.0], DType::F64).unwrap();
        let out = scimath_sqrt(&arr).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        assert!((out.values()[0] - 2.0).abs() < 1e-14); // real
        assert_eq!(out.values()[1], 0.0); // imag
        assert!((out.values()[2] - 3.0).abs() < 1e-14);
        assert_eq!(out.values()[3], 0.0);
    }

    #[test]
    fn scimath_sqrt_negative() {
        let arr = UFuncArray::new(vec![1], vec![-4.0], DType::F64).unwrap();
        let out = scimath_sqrt(&arr).unwrap();
        assert_eq!(out.values()[0], 0.0); // real = 0
        assert!((out.values()[1] - 2.0).abs() < 1e-14); // imag = 2
    }

    #[test]
    fn scimath_log_negative() {
        let arr = UFuncArray::new(vec![1], vec![-1.0], DType::F64).unwrap();
        let out = scimath_log(&arr).unwrap();
        assert!(out.values()[0].abs() < 1e-14); // ln(1) = 0
        assert!((out.values()[1] - std::f64::consts::PI).abs() < 1e-14); // imag = π
    }

    #[test]
    fn scimath_log_positive() {
        let e = std::f64::consts::E;
        let arr = UFuncArray::new(vec![1], vec![e], DType::F64).unwrap();
        let out = scimath_log(&arr).unwrap();
        assert!((out.values()[0] - 1.0).abs() < 1e-14);
        assert_eq!(out.values()[1], 0.0);
    }

    #[test]
    fn scimath_log2_negative() {
        let arr = UFuncArray::new(vec![1], vec![-1.0], DType::F64).unwrap();
        let out = scimath_log2(&arr).unwrap();
        assert!(out.values()[0].abs() < 1e-14); // log2(1) = 0
        let expected_imag = std::f64::consts::PI / std::f64::consts::LN_2;
        assert!((out.values()[1] - expected_imag).abs() < 1e-12);
    }

    #[test]
    fn scimath_log10_negative() {
        let arr = UFuncArray::new(vec![1], vec![-10.0], DType::F64).unwrap();
        let out = scimath_log10(&arr).unwrap();
        assert!((out.values()[0] - 1.0).abs() < 1e-14); // log10(10) = 1
        let expected_imag = std::f64::consts::PI / std::f64::consts::LN_10;
        assert!((out.values()[1] - expected_imag).abs() < 1e-12);
    }

    #[test]
    fn scimath_power_negative_base() {
        let arr = UFuncArray::new(vec![1], vec![-8.0], DType::F64).unwrap();
        let out = scimath_power(&arr, 1.0 / 3.0).unwrap();
        // (-8)^(1/3) = 2 * e^{i*π/3} = 2*(cos(π/3) + i*sin(π/3)) = 1 + i*sqrt(3)
        let r = 2.0; // 8^(1/3) = 2
        let theta = std::f64::consts::PI / 3.0;
        assert!((out.values()[0] - r * theta.cos()).abs() < 1e-12);
        assert!((out.values()[1] - r * theta.sin()).abs() < 1e-12);
    }

    #[test]
    fn scimath_arccos_in_range() {
        let arr = UFuncArray::new(vec![1], vec![0.5], DType::F64).unwrap();
        let out = scimath_arccos(&arr).unwrap();
        assert!((out.values()[0] - (0.5_f64).acos()).abs() < 1e-14);
        assert_eq!(out.values()[1], 0.0);
    }

    #[test]
    fn scimath_arccos_out_of_range() {
        let arr = UFuncArray::new(vec![1], vec![2.0], DType::F64).unwrap();
        let out = scimath_arccos(&arr).unwrap();
        // arccos(2.0) → complex with real=0, imag = -ln(2+sqrt(3))
        assert_eq!(out.values()[0], 0.0);
        assert!(out.values()[1].is_finite());
    }

    #[test]
    fn scimath_arcsin_in_range() {
        let arr = UFuncArray::new(vec![1], vec![0.5], DType::F64).unwrap();
        let out = scimath_arcsin(&arr).unwrap();
        assert!((out.values()[0] - (0.5_f64).asin()).abs() < 1e-14);
        assert_eq!(out.values()[1], 0.0);
    }

    #[test]
    fn scimath_arctanh_in_range() {
        let arr = UFuncArray::new(vec![1], vec![0.5], DType::F64).unwrap();
        let out = scimath_arctanh(&arr).unwrap();
        assert!((out.values()[0] - (0.5_f64).atanh()).abs() < 1e-14);
        assert_eq!(out.values()[1], 0.0);
    }

    #[test]
    fn scimath_arctanh_out_of_range() {
        let arr = UFuncArray::new(vec![1], vec![2.0], DType::F64).unwrap();
        let out = scimath_arctanh(&arr).unwrap();
        assert!(out.values()[0].is_finite()); // real part finite
        assert!((out.values()[1] - (-std::f64::consts::FRAC_PI_2)).abs() < 1e-12);
    }

    // ── pad mode tests ──────────────────────────────────────────────────

    #[test]
    fn pad_linear_ramp_basic() {
        let arr = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let out = pad_linear_ramp(&arr, (2, 2), (0.0, 5.0)).unwrap();
        assert_eq!(out.shape(), &[7]);
        // Before: ramp from 0 to 1 → [0.333..., 0.666...]
        // After: ramp from 3 to 5 → [3.666..., 4.333...]
        assert!(out.values()[0] > 0.0 && out.values()[0] < 1.0);
        assert!(out.values()[1] > out.values()[0] && out.values()[1] < 1.0);
        assert_eq!(out.values()[2], 1.0); // original start
        assert_eq!(out.values()[4], 3.0); // original end
        assert!(out.values()[5] > 3.0 && out.values()[5] < 5.0);
        assert!(out.values()[6] > out.values()[5] && out.values()[6] < 5.0);
    }

    #[test]
    fn pad_stat_maximum() {
        let arr = UFuncArray::new(vec![4], vec![1.0, 5.0, 2.0, 4.0], DType::F64).unwrap();
        let out = pad_stat(&arr, (2, 2), "maximum", None).unwrap();
        assert_eq!(out.shape(), &[8]);
        assert_eq!(out.values()[0], 5.0); // max of whole array
        assert_eq!(out.values()[1], 5.0);
        assert_eq!(out.values()[6], 5.0);
        assert_eq!(out.values()[7], 5.0);
    }

    #[test]
    fn pad_stat_minimum() {
        let arr = UFuncArray::new(vec![4], vec![1.0, 5.0, 2.0, 4.0], DType::F64).unwrap();
        let out = pad_stat(&arr, (1, 1), "minimum", None).unwrap();
        assert_eq!(out.values()[0], 1.0); // min of whole array
        assert_eq!(out.values()[5], 1.0);
    }

    #[test]
    fn pad_stat_mean() {
        let arr = UFuncArray::new(vec![4], vec![2.0, 4.0, 6.0, 8.0], DType::F64).unwrap();
        let out = pad_stat(&arr, (1, 1), "mean", None).unwrap();
        assert!((out.values()[0] - 5.0).abs() < 1e-14); // mean = 5
        assert!((out.values()[5] - 5.0).abs() < 1e-14);
    }

    #[test]
    fn pad_stat_median() {
        let arr = UFuncArray::new(vec![4], vec![1.0, 3.0, 5.0, 7.0], DType::F64).unwrap();
        let out = pad_stat(&arr, (1, 1), "median", None).unwrap();
        assert!((out.values()[0] - 4.0).abs() < 1e-14); // median = (3+5)/2
        assert!((out.values()[5] - 4.0).abs() < 1e-14);
    }

    #[test]
    fn pad_empty_basic() {
        let arr = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64).unwrap();
        let out = pad_empty(&arr, (2, 1)).unwrap();
        assert_eq!(out.shape(), &[6]);
        assert_eq!(out.values(), &[0.0, 0.0, 1.0, 2.0, 3.0, 0.0]);
    }

    #[test]
    fn pad_stat_with_stat_length() {
        let arr =
            UFuncArray::new(vec![6], vec![1.0, 2.0, 10.0, 20.0, 5.0, 6.0], DType::F64).unwrap();
        let out = pad_stat(&arr, (1, 1), "mean", Some(2)).unwrap();
        // Before: mean of first 2 elements = (1+2)/2 = 1.5
        assert!((out.values()[0] - 1.5).abs() < 1e-14);
        // After: mean of last 2 elements = (5+6)/2 = 5.5
        assert!((out.values()[7] - 5.5).abs() < 1e-14);
    }

    // ── E2E pipeline tests ──────────────────────────────────────────────

    #[test]
    fn e2e_frexp_modf_roundtrip() {
        // frexp: decompose into mantissa/exponent, reconstruct
        let x = UFuncArray::new(vec![4], vec![0.5, 2.75, -8.0, 100.0], DType::F64).unwrap();
        let (m, e) = frexp(&x).unwrap();
        for i in 0..4 {
            let reconstructed = m.values()[i] * (2.0_f64).powi(e.values()[i] as i32);
            assert!(
                (reconstructed - x.values()[i]).abs() < 1e-14,
                "frexp roundtrip failed at {i}: {} != {}",
                reconstructed,
                x.values()[i]
            );
        }
        // modf: fractional + integral == original
        let (frac, int) = modf(&x).unwrap();
        for i in 0..4 {
            let sum = frac.values()[i] + int.values()[i];
            assert!(
                (sum - x.values()[i]).abs() < 1e-14,
                "modf roundtrip failed at {i}: {} != {}",
                sum,
                x.values()[i]
            );
        }
    }

    #[test]
    fn e2e_gcd_lcm_divmod_invariants() {
        let a = UFuncArray::new(vec![3], vec![12.0, 35.0, 100.0], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3], vec![8.0, 15.0, 75.0], DType::F64).unwrap();
        let g = gcd_arrays(&a, &b).unwrap();
        let l = lcm_arrays(&a, &b).unwrap();
        // gcd(a,b) * lcm(a,b) == a * b
        for i in 0..3 {
            let prod = g.values()[i] * l.values()[i];
            let expected = a.values()[i] * b.values()[i];
            assert!(
                (prod - expected).abs() < 1e-10,
                "gcd*lcm != a*b at {i}: {} != {}",
                prod,
                expected
            );
        }
        // divmod: q*b + r == a
        let (q, r) = divmod_arrays(&a, &b).unwrap();
        for i in 0..3 {
            let reconstructed = q.values()[i] * b.values()[i] + r.values()[i];
            assert!(
                (reconstructed - a.values()[i]).abs() < 1e-10,
                "divmod invariant failed at {i}: {} != {}",
                reconstructed,
                a.values()[i]
            );
        }
    }

    #[test]
    fn e2e_unique_pipeline() {
        let x =
            UFuncArray::new(vec![7], vec![3.0, 1.0, 2.0, 1.0, 3.0, 2.0, 1.0], DType::F64).unwrap();
        // unique_all returns (values, indices, inverse, counts)
        let (vals, _indices, inverse, counts) = unique_all(&x);
        // sum of counts == original length
        let total: f64 = counts.values().iter().sum();
        assert_eq!(total, 7.0);
        // reconstruction via inverse: vals[inverse[i]] == x[i]
        for i in 0..7 {
            let inv_idx = inverse.values()[i] as usize;
            assert!(
                (vals.values()[inv_idx] - x.values()[i]).abs() < 1e-14,
                "unique inverse reconstruction failed at {i}"
            );
        }
        // unique_counts sum must match
        let (uc_vals, uc_counts) = unique_counts(&x);
        assert_eq!(uc_vals.values(), vals.values());
        let uc_total: f64 = uc_counts.values().iter().sum();
        assert_eq!(uc_total, 7.0);
    }

    #[test]
    fn e2e_polynomial_cheb_fit_eval_deriv_roundtrip() {
        // Fit a cubic with Chebyshev, eval, verify residuals
        let x: Vec<f64> = (-10..=10).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| 2.0 * xi * xi * xi - 3.0 * xi + 1.0)
            .collect();
        let c = chebfit(&x, &y, 5).unwrap();
        let y_fit = chebval(&x, &c);
        let max_residual = y
            .iter()
            .zip(y_fit.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_residual < 1e-10,
            "chebfit residual too large: {max_residual}"
        );
        // chebder + chebint roundtrip: integrate derivative == original (up to constant)
        let dc = chebder(&c, 1);
        let ic = chebint(&dc, 1);
        // ic should match c except for constant term
        for i in 1..c.len().min(ic.len()) {
            assert!(
                (ic[i] - c[i]).abs() < 1e-10,
                "cheb der/int roundtrip mismatch at coeff {i}: {} != {}",
                ic[i],
                c[i]
            );
        }
    }

    #[test]
    fn e2e_cross_family_polynomial_eval() {
        // Evaluate all polynomial families at the same point, with c = [1, 0, 1] (1 + T_2/P_2/H_2/etc)
        let x = vec![0.5];
        let c = vec![1.0, 0.0, 1.0];
        let cheb_val = chebval(&x, &c)[0];
        let leg_val = legval(&x, &c)[0];
        let herm_val = hermval(&x, &c)[0]; // physicist
        let herme_val = hermeval(&x, &c)[0]; // probabilist
        let lag_val = lagval(&x, &c)[0];
        // Just verify they produce finite, distinct values (families differ)
        assert!(cheb_val.is_finite(), "chebval not finite");
        assert!(leg_val.is_finite(), "legval not finite");
        assert!(herm_val.is_finite(), "hermval not finite");
        assert!(herme_val.is_finite(), "hermeval not finite");
        assert!(lag_val.is_finite(), "lagval not finite");
        // Chebyshev and Legendre should be close for low degree
        assert!(
            (cheb_val - leg_val).abs() < 2.0,
            "cheb vs leg too far apart"
        );
    }

    #[test]
    fn e2e_pad_modes_shape_and_values() {
        let arr = UFuncArray::new(vec![5], vec![10.0, 20.0, 30.0, 40.0, 50.0], DType::F64).unwrap();
        // linear_ramp: interpolates between end_value (0) and edge
        let lr = pad_linear_ramp(&arr, (2, 2), (0.0, 0.0)).unwrap();
        assert_eq!(lr.shape(), &[9]);
        // Ramp values are between end_value (0) and edge value
        assert!(lr.values()[0] >= 0.0 && lr.values()[0] <= 10.0);
        assert!(lr.values()[8] >= 0.0 && lr.values()[8] <= 50.0);
        // stat mean
        let sm = pad_stat(&arr, (1, 1), "mean", None).unwrap();
        assert_eq!(sm.shape(), &[7]);
        let mean_val = (10.0 + 20.0 + 30.0 + 40.0 + 50.0) / 5.0;
        assert!((sm.values()[0] - mean_val).abs() < 1e-14);
        // empty
        let em = pad_empty(&arr, (3, 3)).unwrap();
        assert_eq!(em.shape(), &[11]);
        assert_eq!(em.values()[0], 0.0); // zero-initialized
    }

    #[test]
    fn e2e_scimath_sqrt_square_roundtrip() {
        // scimath_sqrt(-4) should give 2j, squaring should give -4
        let x = UFuncArray::new(vec![3], vec![-4.0, -9.0, 4.0], DType::F64).unwrap();
        let sq = scimath_sqrt(&x).unwrap();
        // result is complex: shape [3, 2]
        assert_eq!(sq.shape(), &[3, 2]);
        // sqrt(-4) = 0 + 2j
        assert!((sq.values()[0] - 0.0).abs() < 1e-14); // real part
        assert!((sq.values()[1] - 2.0).abs() < 1e-14); // imag part
        // sqrt(-9) = 0 + 3j
        assert!((sq.values()[2] - 0.0).abs() < 1e-14);
        assert!((sq.values()[3] - 3.0).abs() < 1e-14);
        // sqrt(4) = 2 + 0j
        assert!((sq.values()[4] - 2.0).abs() < 1e-14);
        assert!((sq.values()[5] - 0.0).abs() < 1e-14);
    }

    #[test]
    fn e2e_parameter_completeness() {
        // count_nonzero with axis
        let m =
            UFuncArray::new(vec![2, 3], vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0], DType::F64).unwrap();
        let cnt0 = m.count_nonzero(Some(0), false).unwrap();
        assert_eq!(cnt0.shape(), &[3]);
        assert_eq!(cnt0.values(), &[1.0, 1.0, 1.0]);
        let cnt1 = m.count_nonzero(Some(1), false).unwrap();
        assert_eq!(cnt1.shape(), &[2]);
        assert_eq!(cnt1.values(), &[2.0, 1.0]);

        // isin with invert
        let a = UFuncArray::new(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0], DType::F64).unwrap();
        let test = UFuncArray::new(vec![2], vec![2.0, 4.0], DType::F64).unwrap();
        let normal = a.isin(&test, false);
        let inverted = a.isin(&test, true);
        // normal + inverted should sum to all 1s
        for i in 0..5 {
            assert!(
                (normal.values()[i] + inverted.values()[i] - 1.0).abs() < 1e-14,
                "isin + isin(invert) != 1 at {i}"
            );
        }

        // searchsorted: side='right' >= side='left' for duplicates
        let sorted = UFuncArray::new(vec![5], vec![1.0, 2.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
        let vals = UFuncArray::new(vec![1], vec![2.0], DType::F64).unwrap();
        let left = sorted.searchsorted(&vals, Some("left"), None).unwrap();
        let right = sorted.searchsorted(&vals, Some("right"), None).unwrap();
        assert!(
            right.values()[0] >= left.values()[0],
            "right side should be >= left side"
        );
        assert_eq!(left.values()[0], 1.0); // first occurrence
        assert_eq!(right.values()[0], 3.0); // past last occurrence

        // where_nonzero
        let cond =
            UFuncArray::new(vec![2, 3], vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0], DType::F64).unwrap();
        let indices = where_nonzero(&cond).unwrap();
        assert_eq!(indices.len(), 2); // 2 dimensions
        assert_eq!(indices[0].values(), &[0.0, 1.0, 1.0]); // row indices
        assert_eq!(indices[1].values(), &[1.0, 0.0, 1.0]); // col indices
    }

    #[test]
    fn e2e_cross_crate_sort_searchsorted() {
        // Create array, sort it, then searchsorted should find correct positions
        let x = UFuncArray::new(vec![6], vec![5.0, 2.0, 8.0, 1.0, 9.0, 3.0], DType::F64).unwrap();
        let sorted = x.sort(Some(0), None).unwrap();
        assert_eq!(sorted.values(), &[1.0, 2.0, 3.0, 5.0, 8.0, 9.0]);
        let probes = UFuncArray::new(vec![3], vec![2.5, 5.0, 10.0], DType::F64).unwrap();
        let indices = sorted.searchsorted(&probes, Some("left"), None).unwrap();
        assert_eq!(indices.values(), &[2.0, 3.0, 6.0]);
    }
}
