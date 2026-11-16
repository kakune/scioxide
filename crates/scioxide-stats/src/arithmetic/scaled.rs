//! Interpolation and partition shares in extended-range arithmetic.
use core::cmp::Ordering;
use scioxide_core::{NumericalIssue, RealScalar, Scaled, ScaledAccumulator, SciOxideError};

pub(crate) fn encode_components<const N: usize>(
    values: [f64; N],
) -> Result<[Scaled<f64>; N], SciOxideError> {
    let mut result = [Scaled::ZERO; N];
    for (target, value) in result.iter_mut().zip(values) {
        *target = Scaled::from_value(value)?;
    }
    Ok(result)
}

pub(crate) fn count_ratio(
    numerator: usize,
    denominator: usize,
) -> Result<Scaled<f64>, SciOxideError> {
    if denominator == 0 {
        return Err(SciOxideError::numerical_instability(
            NumericalIssue::Breakdown,
            "count ratio denominator is zero",
        ));
    }
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    scioxide_core::Scaled::<f64>::from_value((quotient) as f64)?.add(
        scioxide_core::Scaled::<f64>::from_value((remainder) as f64)?.div(
            scioxide_core::Scaled::<f64>::from_value((denominator) as f64)?,
        )?,
    )
}

/// Returns the correctly rounded `(1 − fraction) left + fraction right`.
///
/// This is the linear interpolation between adjacent order statistics used by
/// empirical quantiles. Both products are exact, so endpoints of opposite sign
/// cancel without adding rounding error.
pub(crate) fn interpolate<S: RealScalar>(
    left: S,
    right: S,
    fraction: f64,
) -> Result<S, SciOxideError> {
    if fraction == 0.0 {
        return Ok(left);
    }
    if fraction == 1.0 {
        return Ok(right);
    }
    let (left, right) = (left.into_f64(), right.into_f64());
    if S::MANTISSA_DIGITS == f64::MANTISSA_DIGITS && fraction == 0.5 {
        // A binary64 midpoint: TwoSum is exact, and endpoints of magnitude at
        // least 2^-1021 keep both halvings exact, leaving one final rounding.
        let (sum, error) = scioxide_core::backend::two_sum(left, right);
        let halvable = |x: f64| x == 0.0 || x.abs() >= 2.0 * f64::MIN_POSITIVE;
        if sum.is_finite() && halvable(left) && halvable(right) {
            return S::try_from_f64_exact(sum * 0.5 + error * 0.5);
        }
    }
    let mut sum = scioxide_core::RangeAccumulator::<f64>::new();
    sum.add(left)?;
    sum.subtract_product(fraction, left)?;
    sum.add_product(fraction, right)?;
    if S::MANTISSA_DIGITS == f64::MANTISSA_DIGITS {
        S::try_from_f64_exact(sum.finish()?)
    } else {
        sum.finish_scaled::<S>()?.round()
    }
}

pub(crate) fn affine_with_fractions<S: RealScalar>(
    left: S,
    right: S,
    left_fraction: Scaled<f64>,
    right_fraction: Scaled<f64>,
) -> Result<S, SciOxideError> {
    affine_scaled(
        scioxide_core::Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(left))?,
        scioxide_core::Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(right))?,
        left_fraction,
        right_fraction,
    )?
    .round_to()
}

/// Returns the compensated `(1 − fraction) left + fraction right` in extended range.
///
/// The difference of the endpoints keeps both parts, so an interpolated
/// midpoint is exact.
pub(crate) fn interpolate_scaled(
    left: Scaled<f64>,
    right: Scaled<f64>,
    fraction: Scaled<f64>,
) -> Result<ScaledAccumulator<f64>, SciOxideError> {
    let (base, other, weight) = if left.abs_cmp(right) != Ordering::Greater {
        (left, right, fraction)
    } else {
        (right, left, Scaled::<f64>::ONE.sub(fraction)?)
    };
    let mut value = ScaledAccumulator::from_scaled(base);
    if weight.is_zero() {
        return Ok(value);
    }
    let mut contrast = ScaledAccumulator::from_scaled(other);
    contrast.sub_scaled(base)?;
    for part in contrast.parts() {
        value.add_scaled(part.mul(weight)?)?;
    }
    Ok(value)
}

pub(crate) fn affine_scaled(
    left: Scaled<f64>,
    right: Scaled<f64>,
    left_fraction: Scaled<f64>,
    right_fraction: Scaled<f64>,
) -> Result<Scaled<f64>, SciOxideError> {
    if left.abs_cmp(right) != Ordering::Greater {
        left.add(right.sub(left)?.mul(right_fraction)?)
    } else {
        right.add(left.sub(right)?.mul(left_fraction)?)
    }
}

/// Complementary partition shares `p = left / total` and `q = right / total`.
///
/// Each share is an unevaluated two-part sum. Count shares with `total ≤ 2^53`
/// keep the division residual as their low part, so a merged center moves by
/// the exact rational share. Larger counts and shares of rounded masses have
/// zero low parts. The leading parts are the rounded shares used by the moment
/// update formulas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Shares {
    pub p: [Scaled<f64>; 2],
    pub q: [Scaled<f64>; 2],
}

impl Shares {
    /// Returns the shares of rounded masses, without low parts.
    pub(crate) fn rounded(p: Scaled<f64>, q: Scaled<f64>) -> Self {
        Self {
            p: [p, Scaled::ZERO],
            q: [q, Scaled::ZERO],
        }
    }

    /// Returns the shares of two disjoint counts of the merged `total`.
    ///
    /// The smaller share is formed first, and its complement subtracts both of
    /// its parts.
    ///
    /// # Errors
    ///
    /// Returns an error if `total` is zero or the extended exponent range is
    /// exceeded.
    #[inline]
    pub(crate) fn counts(left: usize, right: usize, total: usize) -> Result<Self, SciOxideError> {
        if total as u128 <= (1_u128 << 53) {
            let small = count_share(left.min(right) as f64, total as f64);
            let (high, error) = scioxide_core::backend::two_sum(1.0, -small[0]);
            let large = [high, error - small[1]];
            let (p, q) = if right <= left {
                (large, small)
            } else {
                (small, large)
            };
            return Ok(Self {
                p: encode_components(p)?,
                q: encode_components(q)?,
            });
        }
        Ok(if right <= left {
            let q = count_ratio(right, total)?;
            Self::rounded(Scaled::<f64>::ONE.sub(q)?, q)
        } else {
            let p = count_ratio(left, total)?;
            Self::rounded(p, Scaled::<f64>::ONE.sub(p)?)
        })
    }
}

/// Returns the two-part quotient `part / total` of integer counts with `part ≤ total ≤ 2^53`.
///
/// The product of the rounded quotient and `total` is exact, so subtracting it
/// from `part` gives the exact division residual.
#[inline(always)]
pub(super) fn count_share(part: f64, total: f64) -> [f64; 2] {
    let share = part / total;
    let (product, residual) = scioxide_core::backend::two_product(share, total);
    [share, ((part - product) - residual) / total]
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn count_shares_keep_their_division_residuals() {
        use scioxide_core::RangeAccumulator;
        // part - total * (high + low) is the residual of the two-part share.
        let residual = |part: usize, total: usize, [high, low]: [Scaled<f64>; 2]| {
            let mut exact = RangeAccumulator::<f64>::new();
            exact.add(part as f64).unwrap();
            exact
                .subtract_product(total as f64, high.round().unwrap())
                .unwrap();
            exact
                .subtract_product(total as f64, low.round().unwrap())
                .unwrap();
            exact.finish().unwrap().abs()
        };
        for (left, right) in [(1, 2), (2, 1), (5, 2), (1, 6), (3, 3), (0, 7), (1, 1 << 40)] {
            let total = left + right;
            let shares = Shares::counts(left, right, total).unwrap();
            assert!(residual(left, total, shares.p) <= 2.0_f64.powi(-100));
            assert!(residual(right, total, shares.q) <= 2.0_f64.powi(-100));
            // The smaller share is the rounded quotient and the larger one
            // its rounded complement, as the moment polynomials expect.
            let small = left.min(right) as f64 / total as f64;
            let (p, q) = if right <= left {
                (1.0 - small, small)
            } else {
                (small, 1.0 - small)
            };
            assert_eq!(shares.p[0].round().unwrap(), p);
            assert_eq!(shares.q[0].round().unwrap(), q);
            let smaller = if right <= left { shares.q } else { shares.p };
            let expected = count_share(left.min(right) as f64, total as f64);
            assert_eq!(smaller.map(|part| part.round().unwrap()), expected);
        }
        let rounded = Shares::rounded(Scaled::HALF, Scaled::HALF);
        assert!(rounded.p[1].is_zero() && rounded.q[1].is_zero());
    }

    #[test]
    fn interpolation_rounds_once_across_cancelling_endpoints() {
        assert_eq!(interpolate(-f64::MAX, f64::MAX, 0.5).unwrap(), 0.0);
        let (left, right) = (-0.9_f64, 1.1_f64);
        let fraction = 0.45_f64;
        // Exact rational arithmetic on the binary64 inputs, rounded once.
        let mut exact = scioxide_core::RangeAccumulator::<f64>::new();
        exact.add(left).unwrap();
        exact.subtract_product(fraction, left).unwrap();
        exact.add_product(fraction, right).unwrap();
        assert_eq!(
            interpolate(left, right, fraction).unwrap().to_bits(),
            exact.finish().unwrap().to_bits()
        );
        assert_eq!(interpolate(1.0_f32, 2.0, 0.25).unwrap(), 1.25);
        // The midpoint fast path agrees with the exact accumulation.
        for (left, right) in [
            (1.0_f64, 1.0 + f64::EPSILON),
            (-0.9, 1.1),
            (1e300, -3.0),
            (-0.0, 0.0),
        ] {
            let mut exact = scioxide_core::RangeAccumulator::<f64>::new();
            exact.add(left).unwrap();
            exact.subtract_product(0.5, left).unwrap();
            exact.add_product(0.5, right).unwrap();
            assert_eq!(
                interpolate(left, right, 0.5).unwrap().to_bits(),
                exact.finish().unwrap().to_bits()
            );
        }
        assert_eq!(
            interpolate(-0.0_f64, 3.0, 0.0).unwrap().to_bits(),
            (-0.0_f64).to_bits()
        );
    }

    #[test]
    fn affine_retains_tiny_working_fractions_and_extreme_endpoints() {
        let half = Scaled::HALF;
        assert_eq!(
            affine_with_fractions(-f64::MAX, f64::MAX, half, half).unwrap(),
            0.0
        );
        // Exact result is (2 - 2^-23) * 2^-24: the f32 value immediately below 2^-23.
        let fraction = Scaled::from_parts(1.0, -151).unwrap();
        let complement = Scaled::<f64>::ONE.sub(fraction).unwrap();
        let actual = affine_with_fractions(0.0_f32, f32::MAX, complement, fraction).unwrap();
        assert_eq!(actual.to_bits(), (2.0_f32.powi(-23)).to_bits() - 1);
    }
}
