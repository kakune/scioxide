//! Arithmetic used by the statistics kernels: native `f64` and extended-range
//! arithmetic behind one trait, range checks and promotion ([`range`]),
//! compensated centers ([`center`]), unit deviances ([`deviance`]) and
//! extended-range interpolation of order statistics ([`scaled`]).
mod center;
mod deviance;
mod range;
mod scaled;

pub(crate) use center::{
    PRODUCT_CANCELLATION, advanced_center, advanced_center_scaled, center_difference,
    center_difference_sum, centered, counted_center, divided_center, divided_center_scaled,
    exact_centered_products, normalized_deviation, offset_parts, scaled_center, scaled_second,
    shift, shifted_center, sum_mean, weighted_shifted_center,
};
pub(crate) use deviance::{deviance_series, unit_deviance};
pub(crate) use range::{
    RangeSum, native, native_center, native_delta, native_scaled, native_sources, promote,
};
pub(crate) use scaled::{
    Shares, affine_scaled, affine_with_fractions, count_ratio, encode_components, interpolate,
    interpolate_scaled,
};

use scioxide_core::RealScalar;
use scioxide_core::Scaled;
use scioxide_core::SciOxideError;

/// Compensated sum in either native or extended-range arithmetic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Compensated<N: Arithmetic> {
    inner: N::Sum,
}
impl<N: Arithmetic> Compensated<N> {
    pub(crate) fn zero() -> Self {
        Self { inner: N::ZERO_SUM }
    }
    #[inline(always)]
    pub(crate) fn added(mut self, value: N) -> Result<Self, SciOxideError> {
        N::sum_add(&mut self.inner, value)?;
        Ok(self)
    }
    pub(crate) fn merged(mut self, other: Self) -> Result<Self, SciOxideError> {
        N::sum_merge(&mut self.inner, other.inner)?;
        Ok(self)
    }
    pub(crate) fn parts(self) -> [N; 2] {
        N::sum_parts(self.inner)
    }
    #[inline(always)]
    pub(crate) fn total(self) -> Result<N, SciOxideError> {
        N::sum_total(self.inner)
    }
}

#[inline(always)]
pub(crate) fn integer_power<N: Arithmetic>(
    mut base: N,
    mut power: usize,
) -> Result<N, SciOxideError> {
    let mut value = N::number(1.0)?;
    while power > 0 {
        if power & 1 != 0 {
            value = value.mul(base)?;
        }
        power >>= 1;
        if power > 0 {
            base = base.square()?;
        }
    }
    Ok(value)
}

// One polynomial implementation, statically dispatched over bounded native or
// extended-range arithmetic. The native domain is checked before entering it.
pub(crate) trait Arithmetic: Copy {
    /// Scalar precision of the representation.
    type Real: RealScalar;
    type Sum: Copy + core::fmt::Debug + PartialEq;
    const ZERO_SUM: Self::Sum;
    fn sum_add(sum: &mut Self::Sum, value: Self) -> Result<(), SciOxideError>;
    fn sum_merge(sum: &mut Self::Sum, other: Self::Sum) -> Result<(), SciOxideError>;
    fn sum_total(sum: Self::Sum) -> Result<Self, SciOxideError>;
    fn sum_parts(sum: Self::Sum) -> [Self; 2];
    fn number(value: f64) -> Result<Self, SciOxideError>;
    fn from_real(value: Self::Real) -> Result<Self, SciOxideError>;
    /// Round to the representation's scalar.
    fn real(self) -> Result<Self::Real, SciOxideError>;
    fn add(self, other: Self) -> Result<Self, SciOxideError>;
    fn sub(self, other: Self) -> Result<Self, SciOxideError>;
    fn mul(self, other: Self) -> Result<Self, SciOxideError>;
    fn div(self, other: Self) -> Result<Self, SciOxideError>;
    /// Natural logarithm of the positive ratio `self / other`.
    fn ln_ratio(self, other: Self) -> Result<Self::Real, SciOxideError>;
    fn abs_le(self, other: Self) -> bool;
    fn is_zero(self) -> bool;
    fn square(self) -> Result<Self, SciOxideError> {
        self.mul(self)
    }
    /// Returns the compensated sum of pairwise products.
    fn product_sum(
        values: impl Iterator<Item = Result<(Self, Self), SciOxideError>>,
    ) -> Result<Self, SciOxideError> {
        let mut sums = [Compensated::<Self>::zero(); 4];
        for (i, value) in values.enumerate() {
            let (left, right) = value?;
            sums[i % 4] = sums[i % 4].added(left.mul(right)?)?;
        }
        sums.into_iter()
            .try_fold(Compensated::<Self>::zero(), |sum, part| sum.merged(part))?
            .total()
    }
}
impl Arithmetic for f64 {
    fn product_sum(
        values: impl Iterator<Item = Result<(Self, Self), SciOxideError>>,
    ) -> Result<Self, SciOxideError> {
        let mut failure = None;
        let values = values.scan((), |(), pair| match pair {
            Ok((left, right)) => Some([left * right]),
            Err(error) => {
                failure = Some(error);
                None
            }
        });
        let sum = scioxide_core::compensated_reduce(values);
        if let Some(error) = failure {
            return Err(error);
        }
        let [sum] = sum?;
        sum.total()
    }
    type Real = f64;
    type Sum = scioxide_core::CompensatedSum<f64>;
    const ZERO_SUM: Self::Sum = Self::Sum::ZERO;
    #[inline]
    fn sum_add(sum: &mut Self::Sum, value: Self) -> Result<(), SciOxideError> {
        sum.add(value)
    }
    #[inline]
    fn sum_merge(sum: &mut Self::Sum, other: Self::Sum) -> Result<(), SciOxideError> {
        sum.merge(other)
    }
    #[inline]
    fn sum_total(sum: Self::Sum) -> Result<Self, SciOxideError> {
        sum.total()
    }
    fn sum_parts(sum: Self::Sum) -> [Self; 2] {
        sum.parts()
    }
    #[inline]
    fn div(self, other: Self) -> Result<Self, SciOxideError> {
        Ok(self / other)
    }
    #[inline]
    fn abs_le(self, other: Self) -> bool {
        self.abs() <= other.abs()
    }
    #[inline]
    fn number(value: f64) -> Result<Self, SciOxideError> {
        Ok(value)
    }
    #[inline]
    fn from_real(value: f64) -> Result<Self, SciOxideError> {
        Ok(value)
    }
    #[inline]
    fn real(self) -> Result<f64, SciOxideError> {
        Ok(self)
    }
    #[inline]
    fn ln_ratio(self, other: Self) -> Result<f64, SciOxideError> {
        Ok((self / other).ln())
    }
    #[inline]
    fn is_zero(self) -> bool {
        self == 0.0
    }
    #[inline]
    fn add(self, other: Self) -> Result<Self, SciOxideError> {
        Ok(self + other)
    }
    #[inline]
    fn sub(self, other: Self) -> Result<Self, SciOxideError> {
        Ok(self - other)
    }
    #[inline]
    fn mul(self, other: Self) -> Result<Self, SciOxideError> {
        Ok(self * other)
    }
}
impl<S: RealScalar> Arithmetic for Scaled<S> {
    type Real = S;
    type Sum = scioxide_core::ScaledAccumulator<S>;
    const ZERO_SUM: Self::Sum = Self::Sum::ZERO;
    fn sum_add(sum: &mut Self::Sum, value: Self) -> Result<(), SciOxideError> {
        sum.add_scaled(value)
    }
    fn sum_merge(sum: &mut Self::Sum, other: Self::Sum) -> Result<(), SciOxideError> {
        sum.merge(other)
    }
    fn sum_total(sum: Self::Sum) -> Result<Self, SciOxideError> {
        sum.total()
    }
    fn sum_parts(sum: Self::Sum) -> [Self; 2] {
        sum.parts()
    }
    #[inline]
    fn div(self, other: Self) -> Result<Self, SciOxideError> {
        self.div(other)
    }
    #[inline]
    fn abs_le(self, other: Self) -> bool {
        self.abs_cmp(other) != core::cmp::Ordering::Greater
    }
    #[inline]
    fn number(value: f64) -> Result<Self, SciOxideError> {
        Self::from_value(crate::error_helpers::number(value)?)
    }
    #[inline]
    fn from_real(value: S) -> Result<Self, SciOxideError> {
        Self::from_value(value)
    }
    #[inline]
    fn real(self) -> Result<S, SciOxideError> {
        self.round()
    }
    #[inline]
    fn ln_ratio(self, other: Self) -> Result<S, SciOxideError> {
        crate::error_helpers::finite(self.div(other)?.ln_abs()?)
    }
    #[inline]
    fn is_zero(self) -> bool {
        self.is_zero()
    }
    #[inline]
    fn add(self, other: Self) -> Result<Self, SciOxideError> {
        self.add(other)
    }
    #[inline]
    fn sub(self, other: Self) -> Result<Self, SciOxideError> {
        self.sub(other)
    }
    #[inline]
    fn mul(self, other: Self) -> Result<Self, SciOxideError> {
        self.mul(other)
    }
}

/// Returns `ln(1 + t)` for `t > −1` in extended range.
///
/// For `|t| ≤ ε/2` the result is `t` itself, which keeps full relative accuracy
/// even below the `f64` underflow threshold. For `|t| ≤ 1` it is `ln_1p` of the
/// rounded `t`. Beyond that it is the logarithm of the extended sum `1 + t`.
///
/// # Errors
///
/// Returns extended-range errors, or an error if `1 + t` is not positive.
pub(crate) fn scaled_ln_1p(t: Scaled<f64>) -> Result<Scaled<f64>, SciOxideError> {
    if t.abs_le(Scaled::from_value(f64::EPSILON * 0.5)?) {
        Ok(t)
    } else if t.abs_le(Scaled::ONE) {
        Scaled::from_value(t.round()?.ln_1p())
    } else {
        Scaled::from_value(t.add(Scaled::ONE)?.ln_abs()?)
    }
}

/// Returns `ln(x / reference)` for positive extended values.
///
/// When the ratio lies in `[1/2, 3/2]` it uses [`scaled_ln_1p`] of the relative
/// difference.
///
/// # Errors
///
/// Returns extended-range errors, or an error if the ratio is not positive.
pub(crate) fn scaled_log_ratio(
    x: Scaled<f64>,
    reference: Scaled<f64>,
) -> Result<f64, SciOxideError> {
    let relative = x.sub(reference)?.div(reference)?;
    if relative.abs_le(Scaled::HALF) {
        scaled_ln_1p(relative)?.round()
    } else {
        x.div(reference)?.ln_abs()
    }
}

/// Returns `ln(x / reference)` for positive checked values; see [`scaled_log_ratio`].
///
/// # Errors
///
/// Returns extended-range errors, or an error if the ratio is not positive.
pub(crate) fn log_ratio<S: RealScalar>(x: S, reference: S) -> Result<f64, SciOxideError> {
    scaled_log_ratio(
        Scaled::from_value(x.into_f64())?,
        Scaled::from_value(reference.into_f64())?,
    )
}
