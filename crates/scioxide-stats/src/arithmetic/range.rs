//! Range checks for native `f64` statistics kernels and exact promotion to extended range.
//!
//! A native kernel checks the binary exponent of its operands before any
//! arithmetic. Values within `±64` keep differences, products up to fourth
//! powers and sums over `usize` counts normal. Accumulated powers stay within
//! `±256`. A value outside the range selects the extended-range kernel.
//! Promotion converts both compensated parts exactly, so the extended kernel
//! continues the same sum.
use scioxide_core::{CompensatedSum, RealScalar, Scaled, ScaledAccumulator, SciOxideError};

/// Returns whether an `f64` value is zero or has a binary exponent within `±bound`.
#[inline]
pub(crate) fn native(value: f64, bound: i32) -> bool {
    let exponent = ((value.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    value == 0.0 || (-bound..=bound).contains(&exponent)
}

/// Returns the exact `f64` value of an extended component in the [`native`] range.
///
/// Returns `None` when the component lies outside.
#[inline]
pub(crate) fn native_scaled(value: Scaled<f64>, bound: i32) -> Option<f64> {
    if !(-i64::from(bound)..=i64::from(bound)).contains(&value.exponent()) {
        return None;
    }
    value.round().ok()
}

/// Returns both parts of an extended center as native `f64` values.
///
/// The parts are normalized so that the residual is at most half an ulp of the
/// leading part, as [`super::normalized_deviation`] requires. The leading part
/// lies within `±64` and the residual within `±256`.
#[inline]
pub(crate) fn native_center(center: ScaledAccumulator<f64>) -> Option<[f64; 2]> {
    let [high, low] = center.parts();
    let (high, low) =
        scioxide_core::backend::two_sum(native_scaled(high, 64)?, native_scaled(low, 256)?);
    (native(high, 64) && native(low, 256)).then_some([high, low])
}

/// Returns the native difference `right − left` if it lies within `±64`.
///
/// Its powers up to four then stay in the normal range.
#[inline]
pub(crate) fn native_delta<S: RealScalar>(left: S, right: S) -> Option<f64> {
    let delta = right.into_f64() - left.into_f64();
    native(delta, 64).then_some(delta)
}

/// Returns whether every value lies in the native `±64` range, without branches.
pub(crate) fn native_sources(values: impl Iterator<Item = f64>) -> bool {
    values.fold(true, |native_sources, x| native_sources & native(x, 64))
}

/// Converts a native two-part value to extended range exactly.
///
/// The value may be a compensated sum or a center. Both parts are converted
/// without rounding their sum.
///
/// # Errors
///
/// Returns an error if a part is not finite.
#[inline]
pub(crate) fn promote([high, low]: [f64; 2]) -> Result<ScaledAccumulator<f64>, SciOxideError> {
    Ok(ScaledAccumulator::from_parts(
        Scaled::from_value(high)?,
        Scaled::from_value(low)?,
    ))
}

/// Compensated sum that stays native while its products are in range.
///
/// It converts both parts exactly to extended range on the first extended term.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum RangeSum {
    Native(CompensatedSum<f64>),
    Extended(ScaledAccumulator<f64>),
}

impl RangeSum {
    pub(crate) const ZERO: Self = Self::Native(CompensatedSum::ZERO);

    /// Returns the sum in extended range.
    ///
    /// # Errors
    ///
    /// Returns an error if a native part is not finite.
    pub(crate) fn extended(self) -> Result<ScaledAccumulator<f64>, SciOxideError> {
        match self {
            Self::Native(sum) => promote(sum.parts()),
            Self::Extended(sum) => Ok(sum),
        }
    }
    /// Returns the native total, or `None` once the sum is in extended range.
    ///
    /// # Errors
    ///
    /// Returns an error if the native total overflows.
    pub(crate) fn native(self) -> Result<Option<f64>, SciOxideError> {
        match self {
            Self::Native(sum) => sum.total().map(Some),
            Self::Extended(_) => Ok(None),
        }
    }
    /// Rounds both parts to one extended component.
    ///
    /// # Errors
    ///
    /// Returns an error if the extended exponent range is exceeded.
    pub(crate) fn total(self) -> Result<Scaled<f64>, SciOxideError> {
        self.extended()?.total()
    }
    /// Adds an extended term, converting a native sum first.
    ///
    /// # Errors
    ///
    /// Returns an error if the extended exponent range is exceeded. The sum is
    /// unchanged on error.
    pub(crate) fn add_scaled(&mut self, value: Scaled<f64>) -> Result<(), SciOxideError> {
        let mut sum = self.extended()?;
        sum.add_scaled(value)?;
        *self = Self::Extended(sum);
        Ok(())
    }
    /// Adds a product, in native arithmetic while both factors lie within `±256`.
    ///
    /// # Errors
    ///
    /// Returns an error if a factor is not finite or the extended exponent range
    /// is exceeded.
    #[inline]
    pub(crate) fn add_product(&mut self, left: f64, right: f64) -> Result<(), SciOxideError> {
        // Products of normal factors within ±256 and at most usize::MAX
        // additions fit normal binary64, including their residuals.
        if let Self::Native(sum) = self
            && native(left, 256)
            && native(right, 256)
        {
            return sum.add(left * right);
        }
        self.add_scaled(Scaled::from_value(left)?.mul(Scaled::from_value(right)?)?)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn guards_share_one_exponent_frame() {
        for (value, bound) in [(2.0_f64.powi(64), 64), (2.0_f64.powi(-64), 64), (3.0, 1)] {
            assert!(native(value, bound));
            assert_eq!(
                native_scaled(Scaled::from_value(value).unwrap(), bound),
                Some(value)
            );
        }
        for (value, bound) in [
            (2.0_f64.powi(65), 64),
            (2.0_f64.powi(-65), 64),
            (f64::MIN_POSITIVE / 4.0, 1022),
        ] {
            assert!(!native(value, bound));
            assert_eq!(
                native_scaled(Scaled::from_value(value).unwrap(), bound),
                None
            );
        }
        assert!(native(0.0, 0) && native(-0.0, 0));
        assert_eq!(native_delta(1e300_f64, -1e300), None);
        assert_eq!(native_delta(1.0_f32, 3.0), Some(2.0));
    }

    #[test]
    fn native_centers_are_normalized() {
        // Unnormalized accumulator parts become a rounded value and its residual.
        let tiny = 2.0_f64.powi(-60);
        for ([high, low], expected) in [
            ([1.0, 1.0], [2.0, 0.0]),
            (
                [1.0, 0.75 * f64::EPSILON],
                [1.0 + f64::EPSILON, -0.25 * f64::EPSILON],
            ),
            ([tiny, 1.0], [1.0, tiny]),
            ([3.0, 0.0], [3.0, 0.0]),
        ] {
            let center = promote([high, low]).unwrap();
            assert_eq!(native_center(center), Some(expected));
        }
        let wide = promote([2.0_f64.powi(64), 2.0_f64.powi(64)]).unwrap();
        assert_eq!(native_center(wide), None);
    }

    #[test]
    fn range_sums_promote_without_rounding() {
        let mut sum = RangeSum::ZERO;
        sum.add_product(1e16, 1.0).unwrap();
        sum.add_product(1.0, 1.0).unwrap();
        assert!(matches!(sum, RangeSum::Native(_)));
        sum.add_product(1e300, 1e-300).unwrap();
        assert!(matches!(sum, RangeSum::Extended(_)));
        sum.add_product(-1e16, 1.0).unwrap();
        assert_eq!(sum.total().unwrap().round().unwrap(), 2.0);
        assert_eq!(sum.native().unwrap(), None);
        let tiny = 2.0_f64.powi(-60);
        assert_eq!(
            promote([1.0, tiny]).unwrap().parts(),
            [Scaled::ONE, Scaled::from_value(tiny).unwrap()]
        );
    }
}
