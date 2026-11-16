//! Compensated means about a shifted origin, and differences from two-part centers.
//!
//! A native mean shifts every value by an origin with an exact TwoSum split,
//! sums both parts with compensated summation, and keeps the division residual
//! ([`shifted_center`]). Differences from two-part centers cancel the leading
//! parts before the residuals, natively ([`offset_parts`], [`centered`],
//! [`normalized_deviation`]) and in extended range ([`center_difference_sum`]).
//! Merging two partitions moves a center by a two-part share of a two-part
//! difference and keeps the rounding error of the leading product
//! ([`advanced_center`], [`advanced_center_scaled`]). Native callers check their
//! values with [`super::range`] first.
use super::range::{native, promote};
use scioxide_core::backend::{two_product, two_sum};
use scioxide_core::{CompensatedSum, Scaled, ScaledAccumulator, SciOxideError, compensated_reduce};

/// Exact split of `value - origin` into a rounded difference and its residual.
#[inline(always)]
pub(crate) fn shift(value: f64, origin: f64) -> [f64; 2] {
    let (high, low) = two_sum(value, -origin);
    [high, low]
}

/// Merges the two separately summed parts of a shifted sum before division.
fn merged_parts(
    mut high: CompensatedSum<f64>,
    low: CompensatedSum<f64>,
) -> Result<[f64; 2], SciOxideError> {
    high.merge(low)?;
    Ok(high.parts())
}

// Normalize both numerator parts and the division residual before combining
// the origin. Collapsing the shifted sum first would lose the mean of
// [1e16, 1, -1e16], or a sub-ULP center at a large common offset. A two-part
// mass matters when the origin is far from the center: rounding the mass alone
// would move the center by an ULP of the origin's distance.
/// Returns `origin + numerator / mass` as a normalized two-part center.
///
/// The mass parts must be normalized: `mass[1]` is at most half an ulp of `mass[0]`.
pub(crate) fn divided_center(
    origin: f64,
    numerator: [f64; 2],
    [mass, mass_low]: [f64; 2],
) -> Result<[f64; 2], SciOxideError> {
    let [hi, lo] = numerator;
    let quotient = hi / mass;
    let residual = scioxide_core::RealScalar::mul_add(-quotient, mass, hi);
    let mut center = CompensatedSum::from_parts(origin, 0.0)?;
    center.add(quotient)?;
    center.add(residual / mass)?;
    center.add(lo / mass)?;
    if mass_low != 0.0 {
        center.add(-quotient * mass_low / mass)?;
    }
    let [hi, lo] = center.parts();
    let (hi, lo) = two_sum(hi, lo);
    Ok([hi, lo])
}

/// Returns the faithfully rounded mean `sum / mass` of finite two-part sums.
///
/// The exact division residual of the leading part joins the low part before
/// one final rounding.
#[inline]
pub(crate) fn sum_mean([hi, lo]: [f64; 2], mass: f64) -> f64 {
    let quotient = hi / mass;
    let residual = scioxide_core::RealScalar::mul_add(-quotient, mass, hi);
    quotient + (residual + lo) / mass
}

/// Returns the center `origin + (high + low) / mass` of summed [`shift`] parts.
pub(crate) fn shifted_center(
    origin: f64,
    [high, low]: [CompensatedSum<f64>; 2],
    mass: f64,
) -> Result<[f64; 2], SciOxideError> {
    divided_center(origin, merged_parts(high, low)?, [mass, 0.0])
}

/// Returns [`shifted_center`] for a two-part mass, dividing by both of its parts.
pub(crate) fn weighted_shifted_center(
    origin: f64,
    [high, low]: [CompensatedSum<f64>; 2],
    mass: CompensatedSum<f64>,
) -> Result<[f64; 2], SciOxideError> {
    let [mass, mass_low] = mass.parts();
    let (mass, mass_low) = two_sum(mass, mass_low);
    divided_center(origin, merged_parts(high, low)?, [mass, mass_low])
}

/// Returns `origin + numerator / mass` in extended range for a two-part origin and mass.
///
/// It keeps the division residual of the leading numerator part and the
/// first-order contribution of the low part of the mass.
pub(crate) fn divided_center_scaled(
    origin: ScaledAccumulator<f64>,
    numerator: ScaledAccumulator<f64>,
    mass: ScaledAccumulator<f64>,
) -> Result<ScaledAccumulator<f64>, SciOxideError> {
    let [hi, lo] = numerator.parts();
    let [mass, mass_low] = mass.parts();
    let quotient = hi.div(mass)?;
    let residual = hi.difference_of_products(Scaled::ONE, quotient, mass)?;
    let mut center = origin;
    center.add_scaled(quotient)?;
    center.add_scaled(residual.div(mass)?)?;
    center.add_scaled(lo.div(mass)?)?;
    if !mass_low.is_zero() {
        center.sub_scaled(quotient.mul(mass_low)?.div(mass)?)?;
    }
    Ok(center)
}

/// Returns the extended-range center `origin + Σ(x − origin) / mass`.
///
/// Both parts of every difference and the division residual are kept.
pub(crate) fn scaled_center(
    origin: Scaled<f64>,
    values: impl IntoIterator<Item = Result<Scaled<f64>, SciOxideError>>,
    mass: Scaled<f64>,
) -> Result<ScaledAccumulator<f64>, SciOxideError> {
    let base = ScaledAccumulator::from_scaled(origin);
    let mut shifted = ScaledAccumulator::ZERO;
    for value in values {
        shifted.merge(center_difference_sum(
            base,
            ScaledAccumulator::from_scaled(value?),
        )?)?;
    }
    divided_center_scaled(base, shifted, ScaledAccumulator::from_scaled(mass))
}

/// Returns `base + difference * share` for a two-part difference and share.
///
/// The difference is normalized first, so its leading product keeps its exact
/// rounding error. The low cross terms round once. Callers check that the
/// values are in the native range.
///
/// # Errors
///
/// Returns an error if the compensated sum overflows.
#[inline(always)]
pub(crate) fn advanced_center(
    mut base: CompensatedSum<f64>,
    [high, low]: [f64; 2],
    [share, share_low]: [f64; 2],
) -> Result<CompensatedSum<f64>, SciOxideError> {
    let (high, low) = two_sum(high, low);
    let (product, residual) = two_product(high, share);
    base.add(product)?;
    base.add(residual + (high * share_low + low * share))?;
    Ok(base)
}

/// Returns `base + difference / count` for an integer `count ≤ 2^53` with reciprocal `inverse`.
///
/// The quotient of the normalized leading difference keeps its exact division
/// residual, and only the low terms round. Callers check that the values are
/// in the native range.
///
/// # Errors
///
/// Returns an error if the compensated sum overflows.
#[inline(always)]
pub(crate) fn counted_center(
    mut base: CompensatedSum<f64>,
    [high, low]: [f64; 2],
    count: f64,
    inverse: f64,
) -> Result<CompensatedSum<f64>, SciOxideError> {
    let (high, low) = two_sum(high, low);
    let quotient = high * inverse;
    let (product, error) = two_product(quotient, count);
    // `product` is within a factor two of `high`, so the subtraction is exact.
    let residual = (high - product) - error;
    base.add(quotient)?;
    base.add((residual + low) * inverse)?;
    Ok(base)
}

/// Returns `base + difference * share` in extended range; see [`advanced_center`].
///
/// Every part of `difference` is multiplied by both share parts, and the
/// leading products keep their exact rounding errors. It also evaluates
/// compensated affine combinations such as `a + p * (b − a)`.
///
/// # Errors
///
/// Returns an error if the extended exponent range is exceeded.
pub(crate) fn advanced_center_scaled(
    mut base: ScaledAccumulator<f64>,
    difference: ScaledAccumulator<f64>,
    [share, share_low]: [Scaled<f64>; 2],
) -> Result<ScaledAccumulator<f64>, SciOxideError> {
    for part in difference.parts() {
        let product = part.mul(share)?;
        base.add_scaled(product)?;
        base.add_scaled(part.difference_of_products(share, product, Scaled::ONE)?)?;
        if !share_low.is_zero() {
            base.add_scaled(part.mul(share_low)?)?;
        }
    }
    Ok(base)
}

/// Returns `right − left` of two-part values as a compensated sum.
///
/// The leading parts cancel first. Nonzero residuals are added afterwards.
#[inline(always)]
pub(crate) fn offset_parts(
    [left_high, left_low]: [f64; 2],
    [right_high, right_low]: [f64; 2],
) -> Result<CompensatedSum<f64>, SciOxideError> {
    let mut difference = CompensatedSum::from_parts(right_high, 0.0)?;
    difference.add(-left_high)?;
    if right_low != 0.0 {
        difference.add(right_low)?;
    }
    if left_low != 0.0 {
        difference.add(-left_low)?;
    }
    Ok(difference)
}

/// Relative cancellation threshold `√ε = 2^−26` for `f64`.
///
/// Below it, a rounded sum of centered products is recomputed with exact
/// products. Callers scale it by a bound on the sum of absolute products,
/// such as the Cauchy–Schwarz bound.
pub(crate) const PRODUCT_CANCELLATION: f64 = 1.0 / 67_108_864.0;

/// Returns the correctly rounded sum of `(x − cx)(y − cy)` over `pairs`.
///
/// Both centers are two-part values. Each difference keeps its compensated
/// parts, and every product of parts enters an exact product sum.
///
/// # Errors
///
/// Returns an error if a difference or the rounded sum overflows.
pub(crate) fn exact_centered_products(
    pairs: impl Iterator<Item = (f64, f64)>,
    [left_center, right_center]: [[f64; 2]; 2],
) -> Result<f64, SciOxideError> {
    let mut exact = scioxide_core::ProductSum::<f64>::new();
    for (x, y) in pairs {
        let left = offset_parts(left_center, [x, 0.0])?.parts();
        let right = offset_parts(right_center, [y, 0.0])?.parts();
        for a in left {
            for b in right {
                exact.add_product(a, b)?;
            }
        }
    }
    exact.finish()
}

/// Returns `value − center` for a two-part center, cancelling before rounding.
///
/// The caller has checked that the value and the center parts are in the native range.
#[inline(always)]
pub(crate) fn centered(value: f64, center: [f64; 2]) -> Result<f64, SciOxideError> {
    offset_parts(center, [value, 0.0])?.total()
}

/// Returns `value − center` for a normalized two-part center and a checked native value.
#[inline(always)]
pub(crate) fn normalized_deviation(value: f64, [origin, offset]: [f64; 2]) -> f64 {
    // |offset| <= half an ULP of origin. Near the center Sterbenz's lemma
    // makes value-origin exact; away from it the offset cannot cause large
    // cancellation. The source exponent guard keeps both operations normal.
    (value - origin) - offset
}

/// Returns `right − left` of two extended-range centers, rounded to one part.
///
/// Large common offsets cancel before the low parts are added.
pub(crate) fn center_difference(
    left: ScaledAccumulator<f64>,
    right: ScaledAccumulator<f64>,
) -> Result<Scaled<f64>, SciOxideError> {
    center_difference_sum(left, right)?.total()
}

/// Returns `right − left` of two extended-range centers as a two-part sum.
pub(crate) fn center_difference_sum(
    left: ScaledAccumulator<f64>,
    right: ScaledAccumulator<f64>,
) -> Result<ScaledAccumulator<f64>, SciOxideError> {
    let [left_high, left_low] = left.parts();
    let [right_high, right_low] = right.parts();
    let mut difference = ScaledAccumulator::from_scaled(right_high);
    difference.sub_scaled(left_high)?;
    difference.add_scaled(right_low)?;
    difference.sub_scaled(left_low)?;
    Ok(difference)
}

type SecondSummary = (ScaledAccumulator<f64>, Scaled<f64>);

/// Computes the mean and mean squared deviation in two native compensated passes.
///
/// Returns `None` if a value is missing or out of the native range.
fn second_by(
    n: usize,
    value: impl Fn(usize) -> Option<f64>,
) -> Result<Option<SecondSummary>, SciOxideError> {
    if n == 0 {
        return Ok(None);
    }
    let Some(origin) = value(0) else {
        return Ok(None);
    };
    let mut admitted = true;
    let shifted = compensated_reduce((0..n).map(|i| {
        let x = value(i);
        admitted &= x.is_some();
        shift(x.unwrap_or(0.0), origin)
    }));
    let Ok(shifted) = shifted else {
        return Ok(None);
    };
    if !admitted {
        return Ok(None);
    }
    let center = shifted_center(origin, shifted, n as f64)?;
    let squares = compensated_reduce((0..n).map(|i| {
        let x = value(i);
        let delta = normalized_deviation(x.unwrap_or(0.0), center);
        admitted &= x.is_some() & native(delta, 128);
        [delta * delta]
    }));
    let Ok([squares]) = squares else {
        return Ok(None);
    };
    if !admitted {
        return Ok(None);
    }
    Ok(Some((
        promote(center)?,
        Scaled::from_value(squares.total()? / n as f64)?,
    )))
}

/// Computes the mean and mean squared deviation of a sample that can be read twice.
///
/// If every `native` value is in range, it uses the [`second_by`] passes.
/// Otherwise it restarts in extended arithmetic, with the smallest-magnitude
/// `value` as the origin.
pub(crate) fn scaled_second(
    n: usize,
    native: impl Fn(usize) -> Option<f64>,
    value: impl Fn(usize) -> Result<Scaled<f64>, SciOxideError>,
) -> Result<SecondSummary, SciOxideError> {
    if let Some(summary) = second_by(n, native)? {
        return Ok(summary);
    }
    let mut origin = None::<Scaled<f64>>;
    for i in 0..n {
        let x = value(i)?;
        if origin.is_none_or(|o| x.abs_cmp(o).is_lt()) {
            origin = Some(x);
        }
    }
    let origin = origin.ok_or_else(|| {
        crate::error_helpers::invalid(
            scioxide_core::argument!("values"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "empty sample",
        )
    })?;
    let count = Scaled::from_value(n as f64)?;
    let mean = scaled_center(origin, (0..n).map(&value), count)?;
    let mut squares = ScaledAccumulator::ZERO;
    for i in 0..n {
        squares.add_scaled(
            center_difference(mean, ScaledAccumulator::from_scaled(value(i)?))?.square()?,
        )?;
    }
    Ok((mean, squares.total()?.div(count)?))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::arithmetic::Shares;
    use crate::arithmetic::scaled::count_share;

    #[test]
    fn advanced_centers_keep_both_contrast_and_share_parts() {
        // (1 - 2^65) / 3 moved from 2^65: the leading product rounds, and
        // both the contrast's low part and the share's residual matter.
        let large = 2.0_f64.powi(65);
        // 2^65 + (1 - 2^65) / 3 = (2^66 + 1) / 3, to second order in 2^65.
        let error = |[high, low]: [f64; 2]| {
            let mut exact = scioxide_core::RangeAccumulator::<f64>::new();
            exact.add_product(3.0, high).unwrap();
            exact.add_product(3.0, low).unwrap();
            exact.add(-(2.0 * large)).unwrap();
            exact.add(-1.0).unwrap();
            exact.finish().unwrap().abs()
        };
        let base = CompensatedSum::from_parts(large, 0.0).unwrap();
        let difference = offset_parts([large, 0.0], [1.0, 0.0]).unwrap().parts();
        let center = advanced_center(base, difference, count_share(1.0, 3.0)).unwrap();
        assert!(error(center.parts()) <= 2.0_f64.powi(-36));
        // Dropping the share's low part loses the unit contrast entirely.
        let rounded = advanced_center(base, difference, [1.0 / 3.0, 0.0]).unwrap();
        assert!(error(rounded.parts()) > 0.5);
        let shares = Shares::counts(2, 1, 3).unwrap();
        let extended = advanced_center_scaled(
            promote([large, 0.0]).unwrap(),
            promote(difference).unwrap(),
            shares.q,
        )
        .unwrap();
        assert!(error(extended.parts().map(|part| part.round().unwrap())) <= 2.0_f64.powi(-36));
    }

    #[test]
    fn compensated_masses_divide_far_origins_exactly() {
        // Center 1 at origin 1e15 with mass 3 + 2^-50: the numerator
        // (1 - 1e15) * mass is exact in two parts. Dropping the mass's low
        // part alone would move the center by about 0.3.
        let mass = [3.0, 2.0_f64.powi(-50)];
        let numerator = [(1.0 - 1e15) * 3.0, (1.0 - 1e15) * mass[1]];
        let [high, low] = divided_center(1e15, numerator, mass).unwrap();
        assert!((high - 1.0).abs() + low.abs() < 1e-12);
        let [high, low] = divided_center(1e15, numerator, [mass[0], 0.0]).unwrap();
        assert!((high + low - 1.0).abs() > 0.25);
    }
}
