//! Arithmetic with an extended binary exponent.
//!
//! Values are stored as a significand and an `i64` binary exponent, so
//! intermediate results neither overflow nor underflow. Results are rounded and
//! checked only when converted back to an IEEE type. `ScaledAccumulator` adds a
//! Neumaier correction in the same range, so cancellation can recover terms that
//! a larger partial sum absorbed.
//!
//! The identities are associated constants. Values are built with `from_value`,
//! `from_parts` and `from_log`. `round` allows gradual underflow, while
//! `try_value` rejects a nonzero component that rounds to zero.

#[cfg(test)]
use crate::Complex;
use crate::{RealScalar, SciOxideError, Tolerance};

mod accumulator;
mod real;
mod value;
pub use accumulator::ScaledAccumulator;
pub use value::ScaledValue;
pub(crate) mod rounding;
pub use real::Scaled;
use real::Wide;

/// Splits off the largest exponent of the nonzero values as a common power of two.
///
/// Returns the values divided by `2^e` and `e`, where `e` is the largest exponent
/// of a nonzero value. Zeros keep their sign; an empty or all-zero array uses
/// `e = 0`.
///
/// # Errors
///
/// - `NumericalInstability` (`ExponentOutOfRange`) if a relative exponent leaves
///   the `i64` range.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Scaled, backend::scaled::factor_common_scale};
///
/// let a = Scaled::from_parts(1.5_f64, 2000)?;
/// let (parts, exponent) = factor_common_scale([a, a])?;
/// assert_eq!(exponent, 2000);
/// assert_eq!(parts[0].round()?, 1.5);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn factor_common_scale<S: RealScalar, const N: usize>(
    values: [Scaled<S>; N],
) -> Result<([Scaled<S>; N], i64), SciOxideError> {
    let common = values
        .iter()
        .filter(|v| !v.is_zero())
        .map(|v| v.exponent())
        .max()
        .unwrap_or(0);
    let mut projected = values;
    for value in &mut projected {
        *value = Wide::from(*value).shifted(-i128::from(common)).finish()?;
    }
    Ok((projected, common))
}

/// Returns `absolute + relative · scale` for a [`Tolerance`](crate::Tolerance), without overflow.
///
/// `scale` may be negative; pass its absolute value for a magnitude threshold.
///
/// # Errors
///
/// - `NumericalInstability` (`ExponentOutOfRange`) if the exponent leaves the
///   `i64` range.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Scaled, Tolerance, backend::scaled::scaled_tolerance_threshold};
///
/// // 1 + 0.5 · 1 = 1.5
/// let tolerance = Tolerance::new(1.0_f64, 0.5)?;
/// let threshold = scaled_tolerance_threshold(&tolerance, Scaled::ONE)?;
/// assert_eq!(threshold.round()?, 1.5);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn scaled_tolerance_threshold<S: RealScalar>(
    tolerance: &Tolerance<S>,
    scale: Scaled<S>,
) -> Result<Scaled<S>, SciOxideError> {
    Wide::from(scale)
        .mul(Scaled::from_value(tolerance.relative())?.into())?
        .add(Scaled::from_value(tolerance.absolute())?.into())?
        .finish()
}

/// Returns whether `|magnitude| ≤ absolute + relative · |scale|`.
///
/// # Errors
///
/// - `NumericalInstability` (`ExponentOutOfRange`) if the threshold's exponent
///   leaves the `i64` range.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Scaled, Tolerance, backend::scaled::scaled_magnitude_le_tolerance};
///
/// // 0.5 ≤ 0 + 1 · 1
/// let tolerance = Tolerance::new(0.0_f64, 1.0)?;
/// assert!(scaled_magnitude_le_tolerance(Scaled::HALF, &tolerance, Scaled::ONE)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn scaled_magnitude_le_tolerance<S: RealScalar>(
    magnitude: Scaled<S>,
    tolerance: &Tolerance<S>,
    scale: Scaled<S>,
) -> Result<bool, SciOxideError> {
    Ok(magnitude
        .abs()
        .abs_le(scaled_tolerance_threshold(tolerance, scale.abs())?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extended_range_cancellation_recovers_finite_value() {
        let maximum = Scaled::from_value(f64::MAX).unwrap();
        let mut sum = ScaledAccumulator::ZERO;
        sum.add_scaled(maximum).unwrap();
        sum.add_scaled(maximum).unwrap();
        sum.sub_scaled(maximum).unwrap();
        assert_eq!(sum.round().unwrap(), f64::MAX);
    }

    #[test]
    fn compensation_recovers_hidden_addend() {
        let maximum = Scaled::from_value(f64::MAX).unwrap();
        let mut sum = ScaledAccumulator::ZERO;
        sum.add_scaled(maximum).unwrap();
        sum.add_value(1.0).unwrap();
        sum.sub_scaled(maximum).unwrap();
        assert_eq!(sum.round().unwrap(), 1.0);
    }

    #[test]
    fn final_rounding_preserves_subnormals() {
        let tiny = f64::from_bits(1);
        assert_eq!(Scaled::from_value(tiny).unwrap().round().unwrap(), tiny);
        assert_eq!(
            Scaled::from_value(1.0)
                .unwrap()
                .div(Scaled::from_value(f64::MAX).unwrap())
                .unwrap()
                .round()
                .unwrap(),
            1.0 / f64::MAX
        );
    }

    #[test]
    fn common_projection_preserves_representable_f32_ratios() {
        let maximum = Scaled::from_value(f32::MAX).unwrap();
        let denominator = maximum.add(maximum).unwrap();
        let ([numerator, denominator], exponent) =
            factor_common_scale([Scaled::from_value(1.0_f32).unwrap(), denominator]).unwrap();
        let numerator = numerator.shifted(exponent).unwrap();
        let denominator = denominator.shifted(exponent).unwrap();
        assert!(numerator.div(denominator).unwrap().round().unwrap() > 0.0);
    }
}

#[cfg(test)]
mod muller_arithmetic_tests {
    use super::*;

    #[test]
    fn scaled_square_roots_preserve_extreme_components_and_signed_cut() {
        for exponent in [-4000, -1074, -1, 0, 1, 1023, 4000] {
            let value = Scaled::from_value(1.5_f64)
                .unwrap()
                .shifted(exponent)
                .unwrap();
            let root = value.sqrt().unwrap();
            let ratio = root.mul(root).unwrap().div(value).unwrap().round().unwrap();
            assert!((ratio - 1.0).abs() <= 4.0 * f64::EPSILON);
        }
        for z in [
            Complex::new(-f64::MAX, -0.0),
            Complex::new(f64::MAX, f64::MAX),
            Complex::new(1e300, 1e-300),
        ] {
            let value = ScaledValue::from_value(z).unwrap();
            let root = value.sqrt().unwrap();
            let ratio = root.mul(root).unwrap().div(value).unwrap().round().unwrap();
            assert!((ratio.re - 1.0).abs() <= 8.0 * f64::EPSILON);
            assert!(ratio.im.abs() <= 8.0 * f64::EPSILON);
            if z.im.is_sign_negative() {
                assert!(root.imaginary().is_sign_negative());
            }
        }
        let value = Scaled::from_value(f32::MAX).unwrap().shifted(2000).unwrap();
        let root = value.sqrt().unwrap();
        let ratio = root.mul(root).unwrap().div(value).unwrap().round().unwrap();
        assert!((ratio - 1.0).abs() <= 4.0 * f32::EPSILON);
    }

    #[test]
    fn scaled_product_cancellation_retains_low_bits() {
        // Products individually round to one; their exact difference is epsilon squared.
        let epsilon = f64::EPSILON;
        let a = Scaled::from_value(1.0 + epsilon)
            .unwrap()
            .shifted(2000)
            .unwrap();
        let b = Scaled::from_value(1.0 - epsilon).unwrap();
        let c = Scaled::from_value(1.0).unwrap().shifted(2000).unwrap();
        let result = c
            .difference_of_products(Scaled::from_value(1.0).unwrap(), a, b)
            .unwrap();
        assert_eq!(
            result.shifted(-2000).unwrap().round().unwrap(),
            epsilon * epsilon
        );
    }
}
