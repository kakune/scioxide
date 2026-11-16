//! Lower-level building blocks used by the other SciOxide crates.
//!
//! Most users do not need this module. It follows the same semantic-versioning
//! rules as the crate root. It contains:
//!
//! - typed views of scalar values and slices ([`Scalar`], [`ScalarSlice`],
//!   [`RealBackend`], [`components`]), which select `f32`, `f64`, `Complex32`
//!   or `Complex64` without runtime type inspection or pointer casts;
//! - checked size arithmetic and fallible allocation ([`checked_mul`],
//!   [`try_reserve`], [`try_with_capacity`]) and layout checks
//!   ([`checked_strided_span`]);
//! - error-free transformations and short exact reductions ([`two_sum`],
//!   [`two_product`], [`RealProductSum`], [`four_lane_sums`]);
//! - [`validate_then_commit`], which checks every output value before writing any;
//! - products and substitution steps that avoid overflow ([`standard`]) and
//!   common-scale factoring ([`scaled`]);
//! - CPU kernel selection ([`KernelDispatch`]);
//! - per-partition storage and solve permits for executors ([`PartitionSlot`],
//!   [`with_partitioned_solves`]);
//! - Payne–Hanek argument reduction ([`reduce_radians`], [`TWO_OVER_PI_LIMBS`])
//!   and real scalar kernels with a small error enum ([`ScalarKernelError`]);
//! - branch-free elementary functions on fixed-width arrays ([`lanes_exp`],
//!   [`lanes_exp_m1`], [`lanes_ln`], [`lanes_ln_1p`], [`lanes_sin`],
//!   [`lanes_cos`], [`lanes_tanh`], [`lanes_powf`]).
//!
//! [`KernelDispatch`] checks which instruction sets the host CPU supports: at
//! run time with the `std` feature, and from compile-time target features
//! otherwise. A [`crate::ProcessorCapabilities`] value can restrict this
//! choice but never enable an instruction set that the check did not confirm.
//! There is no unchecked override.
//!
//! # Lane kernels
//!
//! A `lanes_*` function evaluates one binary64 elementary function in every
//! lane of a `[f64; W]` array with a straight-line kernel: IEEE arithmetic and
//! bit operations, special values by selection, and no table lookup, branch or
//! fused multiply-add. Every lane runs the same operations, so the loop
//! vectorizes for the target features of the build (SSE2 by default, AVX2 with
//! `-C target-cpu=native` on recent x86-64), and a lane's result depends only
//! on its arguments: it is the same for every `W`, lane position and target,
//! with or without FMA. Results are within 4 ulps of the exact value and can
//! differ from the scalar [`RealScalar`] method in the last ulps. A lane
//! outside the kernel's domain, stated on each function, returns the scalar
//! method's result, including its IEEE results for zeros, infinities and NaN;
//! checking the domain costs one branch per call, not one per lane. The
//! functions never fail and do not check their arguments.

mod components;
pub use components::{component_count, components, components_mut};

/// Common-scale factoring and tolerance tests for extended-range values.
pub mod scaled;
/// Products and substitution steps that avoid overflow and underflow.
pub mod standard;

pub use crate::arithmetic::{
    ProductFactor, RealProductSum, RealValueSum, ShortReductionRange, absolute_difference_le,
    certified_short_sum_products, certified_short_sum_triples, certified_sum_two_products,
    certified_sum_two_products_with, checked_final_rounding, checked_product_sum,
    compensated_indexed, finish_compensated_lanes, four_lane_sums, opposite_signs, two_product,
    two_sum, validate_then_commit,
};
pub use crate::combinatorics::gcd;
pub use crate::execution::PartitionSlot;
pub use crate::kernel::{KernelBackend, KernelDispatch};
pub use crate::layout::{
    checked_strided_span, matrix_layout_is_injective, validate_layout_backing_len,
};
pub use crate::math::dispatch::{
    lanes_cos, lanes_exp, lanes_exp_m1, lanes_ln, lanes_ln_1p, lanes_powf, lanes_sin, lanes_tanh,
};

pub use crate::size::{
    checked_add, checked_mul, checked_next_power_of_two, checked_product, try_copy_slice,
    try_filled, try_reserve, try_with_capacity,
};
pub use crate::solve::{
    PartitionSolveBatch, PartitionSolvePermit, PartitionSolvePermits, PartitionedSolve,
    with_partitioned_solves,
};

use crate::{Complex, RealScalar, ScalarConversionReason, SciOxideError};

/// Binary expansion of `2/π` in 24-bit limbs, most significant limb first.
///
/// With `C` the integer formed by concatenating the 66 limbs, the exact value
/// satisfies `C · 2^−1584 < 2/π < (C + 1) · 2^−1584`. Users of the table must
/// account for this truncation. [`reduce_radians`] uses it for Payne–Hanek
/// argument reduction.
/// ```
/// use scioxide_core::backend::TWO_OVER_PI_LIMBS;
/// assert_eq!(TWO_OVER_PI_LIMBS.len(), 66);
/// assert!(TWO_OVER_PI_LIMBS.iter().all(|&limb| (0..(1 << 24)).contains(&limb)));
/// ```
pub const TWO_OVER_PI_LIMBS: [i32; 66] = crate::math::IPIO2;

/// Reduces a radian argument to a quadrant and a two-part remainder.
///
/// Returns `(q, high, low)` such that `value − q·π/2` is congruent to
/// `high + low` modulo `2π`, with `high + low` in `[−π/4, π/4]`. Only `q`
/// modulo 4 is meaningful, and `low` is a small correction to `high`. If
/// `|value| ≤ π/4` the result is `(0, value, 0)`; otherwise the reduction uses
/// the Payne–Hanek method with [`TWO_OVER_PI_LIMBS`].
///
/// # Errors
///
/// - `NonFinite` if `value` is NaN or infinite.
///
/// ```
/// let (quadrant, high, low) = scioxide_core::backend::reduce_radians(0.25)?;
/// assert_eq!((quadrant, high, low), (0, 0.25, 0.0));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn reduce_radians(value: f64) -> Result<(i32, f64, f64), SciOxideError> {
    if !value.is_finite() {
        return Err(SciOxideError::non_finite(
            crate::NonFiniteOrigin::Input,
            "phase reduction requires a finite argument",
        ));
    }
    Ok(crate::math::reduce_radians(value))
}

mod private {
    pub trait Sealed {}
    pub trait RealSealed {}
}

macro_rules! canonical_scalar_views {
    ($($variant:ident => $ty:ty),+ $(,)?) => {
        /// The four scalar types supported by [`crate::NumericScalar`].
        ///
        /// ```
        /// use scioxide_core::backend::{Scalar, ScalarKind};
        /// assert_eq!(f64::kind(), ScalarKind::F64);
        /// ```
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum ScalarKind { $(#[doc = concat!("`", stringify!($ty), "`.")] $variant),+ }

        /// A scalar value tagged with its type.
        ///
        /// Any bit pattern is allowed, including NaN, infinities, signed zeros
        /// and subnormals. Converting back with `TryFrom` succeeds only for the
        /// same type and never rounds; another type is a `ScalarConversion`
        /// error with [`ScalarConversionReason::Unsupported`](crate::ScalarConversionReason::Unsupported).
        /// ```
        /// use scioxide_core::backend::ScalarValue;
        /// let value: f64 = ScalarValue::F64(-0.0).try_into()?;
        /// assert_eq!(value.to_bits(), (-0.0_f64).to_bits());
        /// # Ok::<(), scioxide_core::SciOxideError>(())
        /// ```
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub enum ScalarValue { $(#[doc = concat!("`", stringify!($ty), "`.")] $variant($ty)),+ }

        /// A shared scalar slice tagged with its element type.
        ///
        /// Converting back with `TryFrom` returns the original slice, with the
        /// same address and length, only for the same type; another type is a
        /// `ScalarConversion` error with
        /// [`ScalarConversionReason::Unsupported`](crate::ScalarConversionReason::Unsupported).
        /// ```
        /// use scioxide_core::backend::ScalarSlice;
        /// let values = [1.0_f32];
        /// let projected: &[f32] = ScalarSlice::F32(&values).try_into()?;
        /// assert_eq!(projected.as_ptr(), values.as_ptr());
        /// # Ok::<(), scioxide_core::SciOxideError>(())
        /// ```
        #[derive(Debug, Clone, Copy)]
        pub enum ScalarSlice<'a> { $(#[doc = concat!("`", stringify!($ty), "`.")] $variant(&'a [$ty])),+ }

        /// A mutable scalar slice tagged with its element type.
        ///
        /// The view holds the original mutable borrow and cannot be cloned.
        /// Converting back with `TryFrom` returns the original slice only for
        /// the same type; another type is a `ScalarConversion` error with
        /// [`ScalarConversionReason::Unsupported`](crate::ScalarConversionReason::Unsupported)
        /// and leaves the elements unchanged.
        /// ```
        /// use scioxide_core::backend::ScalarSliceMut;
        /// let mut values = [1.0_f64];
        /// let projected: &mut [f64] = ScalarSliceMut::F64(&mut values).try_into()?;
        /// projected[0] = 2.0;
        /// assert_eq!(values, [2.0]);
        /// # Ok::<(), scioxide_core::SciOxideError>(())
        /// ```
        ///
        /// ```compile_fail
        /// use scioxide_core::backend::Scalar;
        /// let mut values = [1.0_f64];
        /// let first = f64::slice_mut(&mut values);
        /// let second = f64::slice_mut(&mut values);
        /// drop((first, second));
        /// ```
        #[derive(Debug)]
        pub enum ScalarSliceMut<'a> { $(#[doc = concat!("`", stringify!($ty), "`.")] $variant(&'a mut [$ty])),+ }

        impl ScalarValue {
            /// Returns the type of the value.
            pub const fn kind(self) -> ScalarKind { match self { $(Self::$variant(_) => ScalarKind::$variant),+ } }
        }
        impl ScalarSlice<'_> {
            /// Returns the element type, also for an empty slice.
            pub const fn kind(self) -> ScalarKind { match self { $(Self::$variant(_) => ScalarKind::$variant),+ } }
        }
        impl ScalarSliceMut<'_> {
            /// Returns the element type.
            pub const fn kind(&self) -> ScalarKind { match self { $(Self::$variant(_) => ScalarKind::$variant),+ } }
        }
        $(
            impl TryFrom<ScalarValue> for $ty {
                type Error = SciOxideError;
                #[inline]
                fn try_from(value: ScalarValue) -> Result<Self, Self::Error> {
                    match value {
                        ScalarValue::$variant(value) => Ok(value),
                        other => Err(projection_error(other.kind(), ScalarKind::$variant)),
                    }
                }
            }
            impl<'a> TryFrom<ScalarSlice<'a>> for &'a [$ty] {
                type Error = SciOxideError;
                #[inline]
                fn try_from(value: ScalarSlice<'a>) -> Result<Self, Self::Error> {
                    match value {
                        ScalarSlice::$variant(value) => Ok(value),
                        other => Err(projection_error(other.kind(), ScalarKind::$variant)),
                    }
                }
            }
            impl<'a> TryFrom<ScalarSliceMut<'a>> for &'a mut [$ty] {
                type Error = SciOxideError;
                #[inline]
                fn try_from(value: ScalarSliceMut<'a>) -> Result<Self, Self::Error> {
                    match value {
                        ScalarSliceMut::$variant(value) => Ok(value),
                        other => Err(projection_error(other.kind(), ScalarKind::$variant)),
                    }
                }
            }
        )+
    };
}
canonical_scalar_views! {
    F32 => f32,
    F64 => f64,
    Complex32 => Complex<f32>,
    Complex64 => Complex<f64>,
}

macro_rules! real_scalar {
    ($variant:ident, $ty:ty) => {
        impl private::Sealed for $ty {}
        impl Scalar for $ty {
            #[inline]
            fn kind() -> ScalarKind {
                ScalarKind::$variant
            }
            #[inline]
            fn value(self) -> ScalarValue {
                ScalarValue::$variant(self)
            }
            #[inline]
            fn slice(values: &[Self]) -> ScalarSlice<'_> {
                ScalarSlice::$variant(values)
            }
            #[inline]
            fn slice_mut(values: &mut [Self]) -> ScalarSliceMut<'_> {
                ScalarSliceMut::$variant(values)
            }
            #[inline]
            fn from_value(value: ScalarValue) -> Result<Self, SciOxideError> {
                value.try_into()
            }
        }
    };
}
real_scalar!(F32, f32);
real_scalar!(F64, f64);
impl<S: RealScalar> private::Sealed for Complex<S> {}
impl<S: RealScalar> Scalar for Complex<S> {
    fn kind() -> ScalarKind {
        match S::complex_slice(&[]) {
            ComplexSlice::F32(_) => ScalarKind::Complex32,
            ComplexSlice::F64(_) => ScalarKind::Complex64,
        }
    }
    fn value(self) -> ScalarValue {
        match S::complex_slice(core::slice::from_ref(&self)) {
            ComplexSlice::F32(values) => ScalarValue::Complex32(values[0]),
            ComplexSlice::F64(values) => ScalarValue::Complex64(values[0]),
        }
    }
    fn slice(values: &[Self]) -> ScalarSlice<'_> {
        match S::complex_slice(values) {
            ComplexSlice::F32(values) => ScalarSlice::Complex32(values),
            ComplexSlice::F64(values) => ScalarSlice::Complex64(values),
        }
    }
    fn slice_mut(values: &mut [Self]) -> ScalarSliceMut<'_> {
        match S::complex_slice_mut(values) {
            ComplexSliceMut::F32(values) => ScalarSliceMut::Complex32(values),
            ComplexSliceMut::F64(values) => ScalarSliceMut::Complex64(values),
        }
    }
    fn from_value(value: ScalarValue) -> Result<Self, SciOxideError> {
        let source = value.kind();
        let project =
            |value| S::from_value(value).map_err(|_| projection_error(source, Self::kind()));
        match value {
            ScalarValue::Complex32(value) => Ok(Self::new(
                project(ScalarValue::F32(value.re))?,
                project(ScalarValue::F32(value.im))?,
            )),
            ScalarValue::Complex64(value) => Ok(Self::new(
                project(ScalarValue::F64(value.re))?,
                project(ScalarValue::F64(value.im))?,
            )),
            _ => Err(projection_error(source, Self::kind())),
        }
    }
}

fn projection_error(source: ScalarKind, destination: ScalarKind) -> SciOxideError {
    SciOxideError::scalar_conversion(
        source.into(),
        destination.into(),
        ScalarConversionReason::Unsupported,
    )
}

/// Conversion of a generic scalar to a value or slice tagged with its concrete type.
///
/// Implemented for `f32`, `f64`, `Complex32` and `Complex64` only. Matching on
/// the returned [`ScalarValue`], [`ScalarSlice`] or [`ScalarSliceMut`] selects
/// code for the concrete type without runtime type inspection. Views keep the
/// original address, length and bits.
///
/// ```
/// use scioxide_core::backend::{Scalar, ScalarSliceMut};
/// let mut values = [1.0_f64, 2.0];
/// if let ScalarSliceMut::F64(values) = f64::slice_mut(&mut values) {
///     values[1] = 3.0;
/// }
/// assert_eq!(values, [1.0, 3.0]);
/// ```
pub trait Scalar: private::Sealed + Copy + Send + Sync + 'static {
    /// Returns the concrete type of `Self`.
    fn kind() -> ScalarKind;
    /// Tags one value with its type, without changing its bits.
    fn value(self) -> ScalarValue;
    /// Tags a shared slice with its element type.
    fn slice(values: &[Self]) -> ScalarSlice<'_>;
    /// Tags a mutable slice with its element type.
    fn slice_mut(values: &mut [Self]) -> ScalarSliceMut<'_>;
    /// Extracts a value of type `Self` from a tagged value.
    ///
    /// The conversion never rounds, widens or drops an imaginary part.
    ///
    /// # Errors
    ///
    /// - `ScalarConversion` (`Unsupported`) if `value` holds another type.
    fn from_value(value: ScalarValue) -> Result<Self, SciOxideError>;
}

/// A shared complex slice tagged with its component type.
///
/// ```
/// use scioxide_core::{Complex32, backend::{ComplexSlice, RealBackend}};
/// let values = [Complex32::new(1.0, 2.0)];
/// let ComplexSlice::F32(view) = f32::complex_slice(&values) else { unreachable!() };
/// assert_eq!(view.as_ptr(), values.as_ptr());
/// ```
#[derive(Debug, Clone, Copy)]
pub enum ComplexSlice<'a> {
    /// `f32` components.
    F32(&'a [Complex<f32>]),
    /// `f64` components.
    F64(&'a [Complex<f64>]),
}
/// A mutable complex slice tagged with its component type.
///
/// The view holds the original mutable borrow and cannot be cloned.
///
/// ```
/// use scioxide_core::{Complex64, backend::{ComplexSliceMut, RealBackend}};
/// let mut values = [Complex64::new(1.0, 2.0)];
/// if let ComplexSliceMut::F64(view) = f64::complex_slice_mut(&mut values) {
///     view[0].re = 3.0;
/// }
/// assert_eq!(values[0].re, 3.0);
/// ```
#[derive(Debug)]
pub enum ComplexSliceMut<'a> {
    /// `f32` components.
    F32(&'a mut [Complex<f32>]),
    /// `f64` components.
    F64(&'a mut [Complex<f64>]),
}

/// Operations specific to the real types `f32` and `f64`.
///
/// Implemented for `f32` and `f64` only. Besides typed slice views, it maps
/// random integer words to floating-point values in `[0, 1)`: every word gives a
/// finite value, zero gives `+0`, and nonzero results are normal numbers.
/// ```
/// use scioxide_core::backend::RealBackend;
/// assert_eq!(f64::from_u64_half_open01(0), 0.0);
/// assert!(f32::from_u32_half_open01(u32::MAX) < 1.0);
/// ```
pub trait RealBackend: Scalar + private::RealSealed {
    /// Extracts a `&[Self]` from a tagged slice.
    ///
    /// # Errors
    ///
    /// - `ScalarConversion` (`Unsupported`) if `values` holds another type.
    fn from_scalar_slice(values: ScalarSlice<'_>) -> Result<&[Self], SciOxideError>;
    /// Extracts a `&mut [Self]` from a tagged mutable slice.
    ///
    /// # Errors
    ///
    /// - `ScalarConversion` (`Unsupported`) if `values` holds another type.
    fn from_scalar_slice_mut(values: ScalarSliceMut<'_>) -> Result<&mut [Self], SciOxideError>;

    /// Converts an `f64` with IEEE round-to-nearest; NaN and infinities stay non-finite.
    fn from_f64_ieee(value: f64) -> Self;
    /// Tags a shared complex slice with its component type.
    fn complex_slice(values: &[Complex<Self>]) -> ComplexSlice<'_>;
    /// Tags a mutable complex slice with its component type.
    fn complex_slice_mut(values: &mut [Complex<Self>]) -> ComplexSliceMut<'_>;
    /// Maps the high `p` bits of `value` to the exact multiple of `2^−p` they
    /// encode in `[0, 1)`, with `p = 24` for `f32` and `p = 53` for `f64`.
    fn from_u64_half_open01(value: u64) -> Self;
    /// Maps the high 24 bits (`f32`) or all 32 bits (`f64`) of `value` to `[0, 1)`.
    ///
    /// Every result is the image of equally many input words.
    fn from_u32_half_open01(value: u32) -> Self;
}

macro_rules! real_backend {
    ($ty:ty, $variant:ident, $shift64:literal, $shift32:literal, $scale64:expr, $scale32:expr) => {
        impl private::RealSealed for $ty {}
        impl RealBackend for $ty {
            fn from_scalar_slice(values: ScalarSlice<'_>) -> Result<&[Self], SciOxideError> {
                values.try_into()
            }
            fn from_scalar_slice_mut(
                values: ScalarSliceMut<'_>,
            ) -> Result<&mut [Self], SciOxideError> {
                values.try_into()
            }

            #[inline]
            fn from_f64_ieee(value: f64) -> Self {
                value as Self
            }
            #[inline]
            fn complex_slice(values: &[Complex<Self>]) -> ComplexSlice<'_> {
                ComplexSlice::$variant(values)
            }
            #[inline]
            fn complex_slice_mut(values: &mut [Complex<Self>]) -> ComplexSliceMut<'_> {
                ComplexSliceMut::$variant(values)
            }
            #[inline]
            fn from_u64_half_open01(value: u64) -> Self {
                ((value >> $shift64) as Self) * $scale64
            }
            #[inline]
            fn from_u32_half_open01(value: u32) -> Self {
                ((value >> $shift32) as Self) * $scale32
            }
        }
    };
}
real_backend!(
    f32,
    F32,
    40,
    8,
    f32::from_bits(0x3380_0000),
    f32::from_bits(0x3380_0000)
);
real_backend!(
    f64,
    F64,
    11,
    0,
    f64::from_bits(0x3ca0_0000_0000_0000),
    f64::from_bits(0x3df0_0000_0000_0000)
);

/// Converts a `u128` to the nearest value of `S`, rounding ties to even.
///
/// The integer is rounded once, directly to the precision of `S`, so `f32`
/// results are not rounded twice through `f64`. Zero gives `+0`. Use it for
/// lengths such as FFT sizes that need not be exactly representable, for
/// example `2^24 + 1` in `f32`.
///
/// # Errors
///
/// - `ScalarConversion` (`OutOfRange`) if the rounded value overflows `S`.
/// ```
/// use scioxide_core::backend::scalar_from_u128_rounded;
/// assert_eq!(scalar_from_u128_rounded::<f32>((1 << 24) + 1)?, (1 << 24) as f32);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn scalar_from_u128_rounded<S: RealScalar>(value: u128) -> Result<S, SciOxideError> {
    if value == 0 {
        return Ok(S::ZERO);
    }
    // Round half-to-even directly to the destination precision. Going through
    // `as f64` first would round twice for f32 destinations. After rounding,
    // the significand has at most `precision` bits, so the conversion below is
    // exact and only `scalbn` can overflow.
    let significant_bits = u128::BITS - value.leading_zeros();
    let precision = S::MANTISSA_DIGITS;
    let shift = significant_bits.saturating_sub(precision);
    let mut significand = value >> shift;
    if shift != 0 {
        let halfway = 1_u128 << (shift - 1);
        let discarded = value & ((1_u128 << shift) - 1);
        if discarded > halfway || (discarded == halfway && !significand.is_multiple_of(2)) {
            significand += 1;
        }
    }
    let significand = S::try_from_f64_exact(significand as f64)?;
    // The shift is bounded by u128::BITS (128), so it fits in i32.
    let exponent = shift as i32;
    let converted = significand.scalbn(exponent);
    if converted.is_finite() {
        Ok(converted)
    } else {
        Err(SciOxideError::scalar_conversion(
            crate::ConversionType::U128,
            S::kind().into(),
            ScalarConversionReason::OutOfRange,
        ))
    }
}

/// Rounds the exact value of a [`crate::ProductSum`] to an extended-range value.
///
/// Each component keeps its full exponent and sign, and its significand is
/// rounded once, directly to the precision of `S`. The result is an
/// approximation, not a guaranteed enclosure.
///
/// # Errors
///
/// Returns a `ScalarConversion` or exponent-range error if a component cannot
/// be converted.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ProductSum, backend::product_sum_scaled};
/// let mut sum = ProductSum::<f32>::new();
/// sum.add_product(f32::MAX, f32::MAX)?;
/// let extended = product_sum_scaled(&sum)?;
/// assert!(extended.real().exponent() > 127);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn product_sum_scaled<S: crate::NumericScalar>(
    sum: &crate::ProductSum<S>,
) -> Result<crate::ScaledValue<crate::Complex<S::Real>>, SciOxideError> {
    let [real, imaginary] = sum.scaled_components()?;
    Ok(crate::ScaledValue::from_components(real, imaginary))
}

mod elementary;
pub use elementary::*;
